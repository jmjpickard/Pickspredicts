"""Local prediction flow: train (optional) -> features -> predict -> publish UI JSON.

Usage examples:
    .venv/bin/python scripts/run_local_publish.py
    .venv/bin/python scripts/run_local_publish.py --train
    .venv/bin/python scripts/run_local_publish.py --courses Catterick Thurles
    .venv/bin/python scripts/run_local_publish.py --loop --loop-seconds 180
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ = load_dotenv(PROJECT_ROOT / ".env")


def _run(cmd: Sequence[str], env: dict[str, str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _summarise_predictions(predictions_path: Path) -> None:
    if not predictions_path.exists():
        print(f"predictions file missing: {predictions_path}")
        return

    with open(predictions_path) as f:
        races = json.load(f)

    num_races = len(races)
    num_runners = sum(len(r.get("runners", [])) for r in races)
    courses = sorted({str(r.get("course", "")) for r in races if r.get("course")})
    strong = sum(
        1
        for race in races
        for runner in race.get("runners", [])
        if runner.get("verdict") == "Strong value"
    )
    no_odds = sum(
        1
        for race in races
        for runner in race.get("runners", [])
        if runner.get("verdict") == "No odds"
    )

    print("\nPrediction summary")
    print(f"- races: {num_races}")
    print(f"- runners: {num_runners}")
    print(f"- strong value selections: {strong}")
    print(f"- runners with no odds: {no_odds}")
    print(f"- courses: {', '.join(courses) if courses else '(none)'}")


def _load_pipeline_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _scoring_courses(args: argparse.Namespace, env: dict[str, str], config: dict) -> list[str]:
    if args.courses:
        return [str(course).strip() for course in args.courses if str(course).strip()]

    env_override = env.get("SCORING_COURSES", "").strip()
    if env_override:
        return [token.strip() for token in env_override.split(",") if token.strip()]

    configured = config.get("racecards", {}).get("scoring_courses", [])
    return [str(course).strip() for course in configured if str(course).strip()]


def _result_years(config: dict) -> str | None:
    configured_dates = config.get("racecards", {}).get("dates", [])
    years = sorted({str(date)[:4] for date in configured_dates if str(date)[:4]})
    if not years:
        return None
    if len(years) == 1:
        return years[0]
    return f"{years[0]}-{years[-1]}"


def _refresh_scoring_results(args: argparse.Namespace, env: dict[str, str]) -> bool:
    from src.ingest.fetch_results import fetch_course

    config = _load_pipeline_config()
    years = _result_years(config)
    if not years:
        print("\nSkipping results refresh: no racecard dates configured")
        return False

    courses_cfg = config.get("courses", {})
    output_dir = PROJECT_ROOT / config["paths"]["raw_results"]
    requested_courses = _scoring_courses(args, env, config)
    refreshed_any = False

    for requested_course in requested_courses:
        course_key = next(
            (
                key
                for key in courses_cfg
                if str(key).strip().lower() == requested_course.strip().lower()
            ),
            None,
        )
        if course_key is None:
            print(f"\nSkipping results refresh for {requested_course}: course not in config")
            continue

        course_cfg = courses_cfg[course_key]
        course_id = course_cfg.get("id")
        race_type = course_cfg.get("type")
        if course_id is None or not race_type:
            print(f"\nSkipping results refresh for {requested_course}: incomplete course config")
            continue

        path = fetch_course(
            course_name=str(course_key),
            course_id=int(course_id),
            years=years,
            race_type=str(race_type),
            output_dir=output_dir,
            refresh=True,
        )
        if path is None:
            print(f"\nWarning: results refresh failed for {requested_course} {years}")
            continue
        refreshed_any = True

    return refreshed_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local model flow and publish UI JSON")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain model before prediction",
    )
    parser.add_argument(
        "--skip-fetch-racecards",
        action="store_true",
        help="Skip fetching latest racecards",
    )
    parser.add_argument(
        "--skip-fetch-odds",
        action="store_true",
        help="Skip fetching live exchange odds",
    )
    parser.add_argument(
        "--skip-refresh-results",
        action="store_true",
        help="Skip refreshing current-year results for scoring courses before settlement",
    )
    parser.add_argument(
        "--courses",
        nargs="+",
        default=None,
        help="Override scoring courses for racecard parsing (example: --courses Catterick Thurles)",
    )
    parser.add_argument(
        "--publish-path",
        default="site/public/predictions.json",
        help="Destination for UI JSON",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously refresh odds -> features -> predict -> publish",
    )
    parser.add_argument(
        "--loop-seconds",
        type=int,
        default=180,
        help="Seconds between loop iterations (default: 180)",
    )
    parser.add_argument(
        "--racecards-every",
        type=int,
        default=10,
        help=(
            "When looping, refresh racecards every N iterations "
            "(default: 10; with 180s loop ~= every 30 mins)"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="When looping, stop after N iterations (0 = run forever)",
    )
    return parser.parse_args()


def _publish_once(
    args: argparse.Namespace,
    env: dict[str, str],
    python_bin: str,
    include_racecards: bool,
    include_train: bool,
) -> None:
    if include_train:
        _run([python_bin, "-m", "src.pipeline", "--step", "train"], env)

    if include_racecards and not args.skip_fetch_racecards:
        _run([python_bin, "-m", "src.pipeline", "--step", "fetch-racecards"], env)

    if not args.skip_fetch_odds:
        have_creds = all(
            env.get(name)
            for name in ["BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD"]
        )
        if have_creds:
            _run([python_bin, "-m", "src.pipeline", "--step", "fetch-odds"], env)
        else:
            print(
                "\nSkipping fetch-odds: missing BETFAIR_APP_KEY/BETFAIR_USERNAME/"
                "BETFAIR_PASSWORD in environment"
            )

    if not args.skip_refresh_results and _refresh_scoring_results(args, env):
        _run([python_bin, "-m", "src.pipeline", "--step", "normalise"], env)

    _run([python_bin, "-m", "src.pipeline", "--step", "features"], env)
    _run([python_bin, "-m", "src.pipeline", "--step", "predict"], env)

    dst_json = PROJECT_ROOT / args.publish_path
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            python_bin,
            "scripts/settle_predictions.py",
            "--input",
            "data/model/predictions.json",
            "--output",
            args.publish_path,
        ],
        env,
    )

    print(f"\nPublished UI JSON: {dst_json}")
    _summarise_predictions(dst_json)


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    if args.courses:
        env["SCORING_COURSES"] = ",".join(args.courses)
        print(f"Using SCORING_COURSES={env['SCORING_COURSES']}")

    python_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")

    if not args.loop:
        _publish_once(args, env, python_bin, include_racecards=True, include_train=args.train)
        return

    if args.loop_seconds < 30:
        raise SystemExit("--loop-seconds must be >= 30 to avoid API hammering")
    if args.racecards_every < 1:
        raise SystemExit("--racecards-every must be >= 1")

    iteration = 0
    print(
        "Starting live loop "
        f"(loop={args.loop_seconds}s, racecards every {args.racecards_every} iterations)"
    )
    while True:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        include_racecards = (iteration == 1) or (iteration % args.racecards_every == 0)
        include_train = args.train and iteration == 1
        _publish_once(
            args,
            env,
            python_bin,
            include_racecards=include_racecards,
            include_train=include_train,
        )

        if args.max_iterations > 0 and iteration >= args.max_iterations:
            print("\nReached --max-iterations, exiting loop.")
            break
        print(f"\nSleeping {args.loop_seconds}s...")
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()
