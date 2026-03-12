"""Enrich predictions JSON with staged results and compute settlement metrics.

Usage:
    .venv/bin/python scripts/settle_predictions.py
    .venv/bin/python scripts/settle_predictions.py --input data/model/predictions.json --output site/public/predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RACE_META_FIELDS = {
    "race_name": "race_name",
    "race_type": "race_type",
    "race_class": "class",
    "pattern": "pattern",
    "off_time": "off_time",
    "is_handicap": "is_handicap",
    "field_size": "field_size",
    "distance_f": "distance_f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Settle predictions JSON against staged race results")
    parser.add_argument(
        "--input",
        default="site/public/predictions.json",
        help="Predictions JSON to enrich",
    )
    parser.add_argument(
        "--output",
        default="site/public/predictions.json",
        help="Destination JSON path",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        predictions: list[dict[str, Any]] = json.load(f)  # type: ignore[type-arg]
    return predictions


def _build_race_meta(races_df: pd.DataFrame, race_ids: set[str]) -> dict[str, dict[str, Any]]:
    filtered = races_df[races_df["race_id"].astype(str).isin(race_ids)].copy()
    meta: dict[str, dict[str, Any]] = {}
    for _, row in filtered.iterrows():
        race_id = str(row["race_id"])
        meta[race_id] = {}
        for out_field, src_field in RACE_META_FIELDS.items():
            if src_field not in row.index:
                continue
            value = row[src_field]
            if pd.isna(value):
                meta[race_id][out_field] = None
            elif out_field == "field_size":
                meta[race_id][out_field] = int(value)
            elif out_field == "distance_f":
                meta[race_id][out_field] = float(value)
            elif out_field == "is_handicap":
                meta[race_id][out_field] = bool(value)
            else:
                meta[race_id][out_field] = str(value)
    return meta


def _build_runner_results(
    runners_df: pd.DataFrame,
    race_ids: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    filtered = runners_df[runners_df["race_id"].astype(str).isin(race_ids)].copy()
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in filtered.iterrows():
        race_id = str(row["race_id"])
        horse_id = str(row["horse_id"])
        finish_position = int(row["finish_position"]) if pd.notna(row.get("finish_position")) else None
        sp_decimal = float(row["sp_decimal"]) if pd.notna(row.get("sp_decimal")) else None
        official_rating = int(row["official_rating"]) if pd.notna(row.get("official_rating")) else None
        results[(race_id, horse_id)] = {
            "finish_position": finish_position,
            "sp_decimal": sp_decimal,
            "official_rating": official_rating,
        }
    return results


def _sort_key(race: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(race.get("date") or ""),
        str(race.get("off_time") or ""),
        str(race.get("race_id") or ""),
    )


def _place_terms(field_size: int, is_handicap: bool) -> tuple[int, float]:
    if is_handicap and field_size >= 16:
        return 4, 1 / 5
    return 3, 1 / 4


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def settle_predictions(
    predictions: list[dict[str, Any]],
    race_meta: dict[str, dict[str, Any]],
    runner_results: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched = sorted(predictions, key=_sort_key)
    cumulative_win_pl = 0.0
    cumulative_ew_pl = 0.0
    settled_races = 0
    settled_bets = 0
    top_pick_wins = 0

    for race in enriched:
        race_id = str(race["race_id"])
        meta = race_meta.get(race_id, {})
        for field, value in meta.items():
            if field not in race or race.get(field) in (None, ""):
                race[field] = value

        field_size = _safe_int(race.get("field_size")) or len(race.get("runners", []))
        is_handicap = bool(race.get("is_handicap") or False)
        num_places, place_fraction = _place_terms(field_size, is_handicap)

        race_has_results = False
        race_win_pl = 0.0
        race_ew_pl = 0.0
        race_bets = 0

        top_pick: dict[str, Any] | None = None
        for runner in race.get("runners", []):
            if top_pick is None or float(runner.get("win_prob") or 0.0) > float(top_pick.get("win_prob") or 0.0):
                top_pick = runner

            runner_key = (race_id, str(runner["horse_id"]))
            result = runner_results.get(runner_key, {})

            finish_position = _safe_int(result.get("finish_position"))
            sp_decimal = _safe_float(result.get("sp_decimal"))
            official_rating = _safe_int(result.get("official_rating"))

            runner["finish_position"] = finish_position
            runner["sp_decimal"] = sp_decimal
            runner["official_rating"] = official_rating if official_rating is not None else runner.get("official_rating")
            runner["won"] = finish_position == 1 if finish_position is not None else None
            runner["placed"] = (finish_position is not None and finish_position <= num_places) if finish_position is not None else None
            runner["is_bet"] = runner.get("verdict") == "Strong value"
            runner["win_pl"] = None
            runner["ew_pl"] = None

            if finish_position is not None:
                race_has_results = True

            if runner["is_bet"] and sp_decimal is not None and finish_position is not None:
                race_bets += 1

                win_return = sp_decimal - 1.0 if finish_position == 1 else -1.0
                runner["win_pl"] = round(win_return, 2)
                race_win_pl += win_return

                ew_win_part = 0.5 * (sp_decimal - 1.0) if finish_position == 1 else -0.5
                place_odds = 1.0 + (sp_decimal - 1.0) * place_fraction
                ew_place_part = 0.5 * (place_odds - 1.0) if finish_position <= num_places else -0.5
                ew_return = ew_win_part + ew_place_part
                runner["ew_pl"] = round(ew_return, 2)
                race_ew_pl += ew_return

        if race_has_results:
            settled_races += 1
            settled_bets += race_bets
            if top_pick is not None and top_pick.get("finish_position") == 1:
                top_pick_wins += 1

            cumulative_win_pl += race_win_pl
            cumulative_ew_pl += race_ew_pl

            race["race_win_pl"] = round(race_win_pl, 2)
            race["race_ew_pl"] = round(race_ew_pl, 2)
            race["cumulative_win_pl"] = round(cumulative_win_pl, 2)
            race["cumulative_ew_pl"] = round(cumulative_ew_pl, 2)
            race["num_bets"] = race_bets
            race["num_places_paid"] = num_places
            race["place_fraction"] = f"1/{int(1 / place_fraction)}"
        else:
            race["race_win_pl"] = None
            race["race_ew_pl"] = None
            race["cumulative_win_pl"] = None
            race["cumulative_ew_pl"] = None
            race["num_bets"] = None
            race["num_places_paid"] = None
            race["place_fraction"] = None

    summary = {
        "total_races": len(enriched),
        "settled_races": settled_races,
        "top_pick_wins": top_pick_wins,
        "settled_bets": settled_bets,
        "win_pl": round(cumulative_win_pl, 2),
        "ew_pl": round(cumulative_ew_pl, 2),
        "win_roi": round((cumulative_win_pl / settled_bets) * 100, 1) if settled_bets else None,
        "ew_roi": round((cumulative_ew_pl / settled_bets) * 100, 1) if settled_bets else None,
        "top_pick_accuracy": round((top_pick_wins / settled_races) * 100, 1) if settled_races else None,
    }
    return enriched, summary


def main() -> None:
    args = parse_args()
    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    predictions = _load_predictions(input_path)
    race_ids = {str(race["race_id"]) for race in predictions}

    races_df = pd.read_parquet(PROJECT_ROOT / "data/staged/parquet/races.parquet")
    runners_df = pd.read_parquet(PROJECT_ROOT / "data/staged/parquet/runners.parquet")

    race_meta = _build_race_meta(races_df, race_ids)
    runner_results = _build_runner_results(runners_df, race_ids)
    enriched, summary = settle_predictions(predictions, race_meta, runner_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2, default=_json_default)

    print(f"Written to {output_path}")
    print(f"Settled races: {summary['settled_races']}/{summary['total_races']}")
    top_accuracy = summary["top_pick_accuracy"]
    if top_accuracy is None:
        print("Top-pick accuracy: NA")
    else:
        print(
            f"Top-pick accuracy: {summary['top_pick_wins']}/{summary['settled_races']} "
            f"({top_accuracy:.1f}%)"
        )
    print(f"Strong value bets settled: {summary['settled_bets']}")
    print(f"Win P&L: {summary['win_pl']:+.2f}pts")
    print(f"E/W P&L: {summary['ew_pl']:+.2f}pts")
    win_roi = summary["win_roi"]
    ew_roi = summary["ew_roi"]
    print(f"Win ROI: {win_roi:+.1f}%" if win_roi is not None else "Win ROI: NA")
    print(f"E/W ROI: {ew_roi:+.1f}%" if ew_roi is not None else "E/W ROI: NA")


if __name__ == "__main__":
    main()
