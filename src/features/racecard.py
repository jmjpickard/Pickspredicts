"""Racecard JSON → DataFrame for scoring mode.

Parses racecard JSON files (from rpscrape/racecards/) into a DataFrame
matching the runners schema, so feature groups can compute features for
upcoming race entries.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Furlong to meters conversion
FURLONG_TO_METERS = 201.168


def load_config() -> dict:  # type: ignore[type-arg]
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _allowed_courses(config: dict) -> set[str]:  # type: ignore[type-arg]
    env_override = os.getenv("SCORING_COURSES", "").strip()
    if env_override:
        env_courses = {
            token.strip().lower()
            for token in env_override.split(",")
            if token.strip()
        }
        if env_courses:
            return env_courses

    racecards_cfg = config.get("racecards", {})
    configured = racecards_cfg.get("scoring_courses")
    if configured and isinstance(configured, list):
        courses = {str(c).strip().lower() for c in configured if str(c).strip()}
        if courses:
            return courses
    # Backward-compatible default behavior
    return {"cheltenham"}


def _flatten_races(data: dict[str, object]) -> list[dict[str, object]]:
    """Flatten nested racecard JSON {region: {course: {time: race}}} into a flat list."""
    races: list[dict[str, object]] = []
    for region_courses in data.values():
        if not isinstance(region_courses, dict):
            continue
        for course_times in region_courses.values():
            if not isinstance(course_times, dict):
                continue
            for race in course_times.values():
                if isinstance(race, dict):
                    races.append(race)
    return races


def parse_racecard(json_path: Path, allowed_courses: set[str]) -> pd.DataFrame:
    """Parse a single racecard JSON file into a runners-compatible DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    rows: list[dict[str, object]] = []

    for race in _flatten_races(data):
        course = str(race.get("course", ""))
        if course.strip().lower() not in allowed_courses:
            continue

        race_id = race.get("race_id", "")
        date = race.get("date", "")
        off_time = race.get("off_time", "")
        race_name = race.get("race_name", "")
        race_type = _normalise_race_type(race.get("race_type"))
        race_class = race.get("race_class", "")
        pattern = race.get("pattern", "")
        going = race.get("going", "")
        is_handicap = 1 if race.get("handicap") else 0
        distance_f = _safe_float(race.get("distance_f"))
        distance_m = int(distance_f * FURLONG_TO_METERS) if distance_f else None

        active_runners = [
            r for r in race.get("runners", [])  # type: ignore[union-attr]
            if not r.get("non_runner")
        ]
        field_size = len(active_runners)

        for runner in active_runners:
            rows.append({
                "race_id": race_id,
                "date": date,
                "course": course,
                "off_time": off_time,
                "horse_id": runner.get("horse_id"),
                "horse_name": runner.get("name", runner.get("horse", "")),
                "age": _safe_int(runner.get("age")),
                "official_rating": _safe_int(runner.get("ofr")),
                "topspeed": _safe_int(runner.get("ts")),
                "jockey_id": runner.get("jockey_id"),
                "jockey": runner.get("jockey", ""),
                "trainer_id": runner.get("trainer_id"),
                "trainer": runner.get("trainer", ""),
                "sire_id": runner.get("sire_id"),
                "sire": runner.get("sire", ""),
                "dam_id": runner.get("dam_id"),
                "dam": runner.get("dam", ""),
                "damsire_id": runner.get("damsire_id"),
                "damsire": runner.get("damsire", ""),
                "headgear": runner.get("headgear", ""),
                "weight_lbs": _safe_int(runner.get("lbs")),
                # Racecard-specific fields
                "days_since_last_run_rc": _safe_int(runner.get("last_run")),
                "first_time_headgear_rc": 1 if runner.get("headgear_first") else 0,
                "is_handicap": is_handicap,
                # Race-level fields (duplicated for enriched view compatibility)
                "race_name": race_name,
                "race_type": race_type,
                "race_class": race_class,
                "pattern": pattern,
                "going": going,
                "distance_f": distance_f,
                "distance_m": distance_m,
                "field_size": field_size,
            })

    df = pd.DataFrame(rows)
    logger.info("Parsed %d runners from %s", len(df), json_path.name)
    return df


def load_racecards() -> pd.DataFrame:
    """Load all racecard JSONs and return a combined DataFrame."""
    config = load_config()
    racecard_dir = PROJECT_ROOT / config["paths"]["raw_racecards"]
    allowed_courses = _allowed_courses(config)

    json_files = sorted(racecard_dir.glob("*.json"))
    if not json_files:
        logger.warning("No racecard JSON files found in %s", racecard_dir)
        return pd.DataFrame()

    frames = [parse_racecard(f, allowed_courses) for f in json_files]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["race_id", "horse_id"])
    logger.info("Total racecard runners: %d", len(combined))
    return combined


def _normalise_race_type(raw: object) -> str | None:
    if raw is None:
        return None
    mapping = {"chase": "chase", "hurdle": "hurdle", "nh flat": "nh_flat", "flat": "flat"}
    return mapping.get(str(raw).strip().lower())


def _safe_float(val: object) -> float | None:
    try:
        if val is None:
            return None
        return float(str(val))
    except (ValueError, TypeError):
        return None


def _safe_int(val: object) -> int | None:
    f = _safe_float(val)
    return int(f) if f is not None else None
