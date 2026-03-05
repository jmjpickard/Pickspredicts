"""Build enriched predictions.json for 2025 retrospective site.

Merges model predictions with actual results, race metadata, and computes
running P&L for 1pt win and 1pt each-way strategies on "Strong value" picks.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Load predictions
    with open(PROJECT_ROOT / "data/model/predictions.json") as f:
        pred_races: list[dict] = json.load(f)  # type: ignore[type-arg]

    # Load staged data for results + metadata
    races_df = pd.read_parquet(PROJECT_ROOT / "data/staged/parquet/races.parquet")
    runners_df = pd.read_parquet(PROJECT_ROOT / "data/staged/parquet/runners.parquet")

    # Filter to Cheltenham 2025
    chelt_races = races_df[
        (races_df["course"] == "Cheltenham")
        & (races_df["date"] >= "2025-03-11")
        & (races_df["date"] <= "2025-03-14")
    ]

    # Build race metadata lookup
    race_meta: dict[str, dict] = {}  # type: ignore[type-arg]
    for _, row in chelt_races.iterrows():
        rid = str(row["race_id"])
        race_meta[rid] = {
            "race_name": str(row["race_name"]) if pd.notna(row["race_name"]) else None,
            "race_type": str(row["race_type"]) if pd.notna(row["race_type"]) else None,
            "race_class": str(row["class"]) if pd.notna(row["class"]) else None,
            "pattern": str(row["pattern"]) if pd.notna(row["pattern"]) else None,
            "off_time": str(row["off_time"]) if pd.notna(row["off_time"]) else None,
            "is_handicap": bool(row["is_handicap"]) if pd.notna(row["is_handicap"]) else False,
            "field_size": int(row["field_size"]) if pd.notna(row["field_size"]) else None,
            "distance_f": float(row["distance_f"]) if pd.notna(row["distance_f"]) else None,
        }

    # Build runner result lookup: (race_id, horse_id) → result
    chelt_runners = runners_df[runners_df["race_id"].isin(chelt_races["race_id"])]
    runner_results: dict[tuple[str, str], dict] = {}  # type: ignore[type-arg]
    for _, row in chelt_runners.iterrows():
        key = (str(row["race_id"]), str(row["horse_id"]))
        runner_results[key] = {
            "finish_position": int(row["finish_position"]) if pd.notna(row["finish_position"]) else None,
            "sp_decimal": float(row["sp_decimal"]) if pd.notna(row["sp_decimal"]) else None,
            "official_rating": int(row["official_rating"]) if pd.notna(row["official_rating"]) else None,
        }

    # Enrich predictions
    cumulative_win_pl = 0.0
    cumulative_ew_pl = 0.0

    # Sort races by date then off_time
    for race in pred_races:
        rid = race["race_id"]
        meta = race_meta.get(rid, {})
        race.update(meta)

    pred_races.sort(key=lambda r: (r.get("date", ""), r.get("off_time", "")))

    enriched_races: list[dict] = []  # type: ignore[type-arg]

    for race in pred_races:
        rid = race["race_id"]
        meta = race_meta.get(rid, {})
        field_size = meta.get("field_size", 0) or 0
        is_handicap = meta.get("is_handicap", False)

        # Cheltenham place terms: 3 places at 1/4 odds
        # Handicaps with 16+ runners: 4 places at 1/5 odds
        if is_handicap and field_size >= 16:
            num_places = 4
            place_fraction = 1 / 5
        else:
            num_places = 3
            place_fraction = 1 / 4

        race_win_pl = 0.0
        race_ew_pl = 0.0
        race_bets = 0

        for runner in race["runners"]:
            hid = runner["horse_id"]
            result = runner_results.get((rid, hid), {})
            runner["finish_position"] = result.get("finish_position")
            runner["sp_decimal"] = result.get("sp_decimal")
            runner["official_rating"] = result.get("official_rating")

            fp = runner["finish_position"]
            sp = runner["sp_decimal"]
            runner["won"] = fp == 1 if fp is not None else None
            runner["placed"] = fp is not None and fp <= num_places

            # P&L for "Strong value" picks only
            runner["is_bet"] = runner["verdict"] == "Strong value"
            runner["win_pl"] = None
            runner["ew_pl"] = None

            if runner["is_bet"] and sp is not None and fp is not None:
                race_bets += 1

                # 1pt win
                if fp == 1:
                    win_return = sp - 1  # profit
                else:
                    win_return = -1.0
                runner["win_pl"] = round(win_return, 2)
                race_win_pl += win_return

                # 1pt each-way (0.5pt win + 0.5pt place)
                # Win part
                if fp == 1:
                    ew_win_part = 0.5 * (sp - 1)
                else:
                    ew_win_part = -0.5

                # Place part
                place_odds = 1 + (sp - 1) * place_fraction
                if fp <= num_places:
                    ew_place_part = 0.5 * (place_odds - 1)
                else:
                    ew_place_part = -0.5

                ew_return = ew_win_part + ew_place_part
                runner["ew_pl"] = round(ew_return, 2)
                race_ew_pl += ew_return

        cumulative_win_pl += race_win_pl
        cumulative_ew_pl += race_ew_pl

        race["race_win_pl"] = round(race_win_pl, 2)
        race["race_ew_pl"] = round(race_ew_pl, 2)
        race["cumulative_win_pl"] = round(cumulative_win_pl, 2)
        race["cumulative_ew_pl"] = round(cumulative_ew_pl, 2)
        race["num_bets"] = race_bets
        race["num_places_paid"] = num_places
        race["place_fraction"] = f"1/{int(1/place_fraction)}"

        enriched_races.append(race)

    # Summary
    total_bets = sum(r["num_bets"] for r in enriched_races)
    print(f"Total races: {len(enriched_races)}")
    print(f"Total 'Strong value' bets: {total_bets}")
    print(f"1pt Win P&L: {cumulative_win_pl:+.2f}pts")
    print(f"1pt E/W P&L: {cumulative_ew_pl:+.2f}pts")
    print(f"Win ROI: {cumulative_win_pl / total_bets * 100:+.1f}%" if total_bets else "No bets")
    print(f"E/W ROI: {cumulative_ew_pl / total_bets * 100:+.1f}%" if total_bets else "No bets")

    # Write
    out_path = PROJECT_ROOT / "site/public/predictions.json"
    with open(out_path, "w") as f:
        json.dump(enriched_races, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
