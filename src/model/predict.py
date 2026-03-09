"""Score runners with trained LightGBM model, compute Harville place probs and value overlay.

Usage:
    python -m src.pipeline --step predict
"""

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.ingest.racecard_health import validate_racecard_files

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_COLS = {
    "race_id",
    "horse_id",
    "date",
    "course",
    "horse_name",
    "finish_position",
    "won",
    "placed",
}

PLAN_A_POINTS_LADDER = [2.0, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0]


def load_config() -> dict[str, Any]:
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _softmax_per_race(df: pd.DataFrame, col: str = "raw_prob") -> pd.Series:  # type: ignore[type-arg]
    """Normalise probabilities within each race so they sum to 1."""
    result: pd.Series[float] = df.groupby("race_id")[col].transform(lambda x: x / x.sum())  # type: ignore[assignment]
    return result


def _harville_place_probs(win_probs: np.ndarray[Any, np.dtype[np.floating[Any]]], num_places: int = 3) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute Harville place probabilities (P of finishing in top num_places).

    Args:
        win_probs: array of win probabilities for all runners in a race (sums to ~1)
        num_places: number of places to compute (default 3)

    Returns:
        array of place probabilities, same length as win_probs
    """
    n = len(win_probs)

    # P(i finishes 1st) = win_prob[i]
    p1 = win_probs.astype(np.float64)

    if num_places >= 2:
        # P(i finishes 2nd) = sum over j!=i of [ p_j * p_i / (1 - p_j) ]
        p2 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                denom = 1.0 - float(win_probs[j])
                if denom > 1e-12:
                    p2[i] += float(win_probs[j]) * (float(win_probs[i]) / denom)
    else:
        p2 = np.zeros(n)

    if num_places >= 3:
        # P(i finishes 3rd) = sum over j!=i, k!=i,k!=j of [ p_j * p_k/(1-p_j) * p_i/(1-p_j-p_k) ]
        p3 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                d1 = 1.0 - float(win_probs[j])
                if d1 < 1e-12:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    d2 = 1.0 - float(win_probs[j]) - float(win_probs[k])
                    if d2 < 1e-12:
                        continue
                    p3[i] += float(win_probs[j]) * (float(win_probs[k]) / d1) * (float(win_probs[i]) / d2)
    else:
        p3 = np.zeros(n)

    return p1 + p2 + p3  # type: ignore[return-value]


def _top_features(
    booster: lgb.Booster, row: pd.DataFrame, feature_cols: list[str], top_n: int = 3
) -> str:
    """Get top N feature contributions for a single runner as a formatted string."""
    contribs = booster.predict(row[feature_cols], pred_contrib=True)
    contribs_array = np.asarray(contribs)
    # pred_contrib returns (1, n_features + 1) — last element is bias
    feature_contribs = contribs_array[0, :-1]
    top_idx = np.argsort(np.abs(feature_contribs))[::-1][:top_n]
    parts: list[str] = []
    for idx in top_idx:
        val = float(feature_contribs[idx])
        sign = "+" if val >= 0 else ""
        parts.append(f"{feature_cols[idx]}({sign}{val:.3f})")
    return ", ".join(parts)


def _fmt_pct(prob: float) -> str:
    return f"{prob * 100:.1f}%"


def _fmt_odds(odds: float) -> str:
    text = f"{odds:.2f}"
    return text.rstrip("0").rstrip(".")


def _value_views(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_odds = group[
        group["best_odds"].notna()
        & (group["best_odds"] > 1)
        & group["value_score"].notna()
    ].copy()
    if with_odds.empty:
        return with_odds, with_odds

    realistic = with_odds[
        (with_odds["value_score"] >= 0.015)
        & (with_odds["best_odds"] <= 30)
        & (with_odds["win_prob"] >= 0.05)
    ].copy()
    realistic = realistic.sort_values("value_score", ascending=False)
    return with_odds, realistic


def _plan_a_score(row: pd.Series) -> float:
    win_prob = float(row.get("win_prob", 0.0))
    value_score = max(float(row.get("value_score", 0.0)), 0.0)
    odds = float(row.get("best_odds", np.nan))
    score = (value_score * 2.0) + (win_prob * 0.8)
    if pd.notna(odds) and odds > 30:
        score -= 0.15
    elif pd.notna(odds) and odds > 20:
        score -= 0.08
    return score


def _select_plan_a_runner(group: pd.DataFrame, strong_thresh: float) -> pd.Series | None:
    if group.empty:
        return None
    ordered = group.sort_values("win_prob", ascending=False)
    with_odds, realistic = _value_views(ordered)

    if not realistic.empty:
        return realistic.sort_values(
            ["value_score", "win_prob"],
            ascending=[False, False],
        ).iloc[0]

    if not with_odds.empty:
        strong = with_odds[
            (with_odds["value_score"] >= strong_thresh)
            & (with_odds["win_prob"] >= 0.12)
        ].copy()
        if not strong.empty:
            return strong.sort_values(
                ["value_score", "win_prob"],
                ascending=[False, False],
            ).iloc[0]

    return None


def _build_plan_a_points(
    out_df: pd.DataFrame,
    strong_thresh: float,
) -> dict[tuple[str, str], float]:
    """Build a 10pt daily Plan A using one primary selection per race."""
    plan_points: dict[tuple[str, str], float] = {}
    if out_df.empty:
        return plan_points

    working = out_df.copy()
    if "date" not in working.columns:
        return plan_points
    working["date"] = working["date"].astype(str).str[:10]

    for date, day_group in working.groupby("date"):
        race_candidates: list[dict[str, object]] = []
        for race_id, race_group in day_group.groupby("race_id"):
            selected = _select_plan_a_runner(race_group, strong_thresh=strong_thresh)
            if selected is None:
                continue
            race_candidates.append({
                "date": str(date),
                "race_id": str(race_id),
                "horse_id": str(selected.get("horse_id")),
                "score": _plan_a_score(selected),
            })

        if not race_candidates:
            continue

        ranked = sorted(race_candidates, key=lambda item: float(item["score"]), reverse=True)
        count = len(ranked)
        if count == 1:
            points = [10.0]
        elif count == 2:
            points = [6.0, 4.0]
        elif count == 3:
            points = [4.0, 3.0, 3.0]
        elif count == 4:
            points = [3.5, 2.5, 2.0, 2.0]
        elif count == 5:
            points = [3.0, 2.0, 2.0, 1.5, 1.5]
        elif count == 6:
            points = [2.5, 2.0, 1.5, 1.5, 1.5, 1.0]
        elif count == 7:
            points = PLAN_A_POINTS_LADDER.copy()
        else:
            points = PLAN_A_POINTS_LADDER + [0.5] * (count - len(PLAN_A_POINTS_LADDER))
            scale = 10.0 / sum(points)
            points = [round(p * scale, 2) for p in points]
            points[0] = round(points[0] + (10.0 - sum(points)), 2)

        for item, pts in zip(ranked, points):
            key = (str(item["race_id"]), str(item["horse_id"]))
            plan_points[key] = float(pts)

    return plan_points


def _build_race_analysis(group: pd.DataFrame, strong_thresh: float) -> str:
    """Build concise race-level narrative with a value-first lens."""
    if group.empty:
        return ""

    group = group.copy()
    group = group.sort_values("win_prob", ascending=False)
    model_top = group.iloc[0]
    model_name = str(model_top.get("horse_name", ""))
    model_wp = float(model_top.get("win_prob", 0.0))

    with_odds, realistic_value = _value_views(group)
    if with_odds.empty:
        return (
            f"{model_name} is top on model win chance ({_fmt_pct(model_wp)}), "
            "but there are no usable live odds yet."
        )

    market_fav = with_odds.sort_values("best_odds", ascending=True).iloc[0]
    value_top = with_odds.sort_values("value_score", ascending=False).iloc[0]

    market_name = str(market_fav.get("horse_name", ""))
    value_name = str(value_top.get("horse_name", ""))
    value_score = float(value_top.get("value_score", 0.0))

    if value_score < 0:
        return (
            f"{model_name} leads on win%, but this race looks mostly fair-priced."
        )

    if value_score < 0.02 and realistic_value.empty:
        return (
            "Front of market is tight/fair; little clean value except very speculative outsiders."
        )

    if not realistic_value.empty:
        top_realistic = realistic_value.iloc[0]
        top_realistic_name = str(top_realistic.get("horse_name", ""))
        top_realistic_vs = float(top_realistic.get("value_score", 0.0))
        if (
            len(realistic_value) >= 2
            and top_realistic_vs >= 0.02
            and float(realistic_value.iloc[1].get("value_score", 0.0)) >= 0.02
        ):
            second_name = str(realistic_value.iloc[1].get("horse_name", ""))
            return f"Best priced edges are {top_realistic_name} and {second_name}."
        if top_realistic_name == model_name == market_name and top_realistic_vs >= strong_thresh:
            return f"{top_realistic_name} is both top win% and genuine overlay."
        if top_realistic_name != market_name and float(market_fav.get("value_score", -1.0)) < 0:
            return f"Market fav looks short; {top_realistic_name} is best value play."
        if top_realistic_name != market_name:
            if top_realistic_vs >= 0.04 and float(top_realistic.get("win_prob", 0.0)) >= 0.10:
                return f"{top_realistic_name} is the strongest non-fav value."
            return f"{top_realistic_name} has better value than fav {market_name}."

    if value_name != market_name:
        return (
            f"{value_name} has better value than fav {market_name}."
        )

    if value_name == model_name == market_name and value_score >= strong_thresh:
        return f"{value_name} is both top win% and genuine overlay."

    return f"{model_name} is top win%, but edge versus price is modest."


def predict() -> None:
    """Score runners, compute Harville place probs, value overlay, and save predictions."""
    config = load_config()
    model_cfg = config["model"]
    output_dir = PROJECT_ROOT / model_cfg["output_dir"]
    racecard_dir = PROJECT_ROOT / config["paths"]["raw_racecards"]
    racecard_files = validate_racecard_files(racecard_dir, config)
    latest_racecard_mtime = max(path.stat().st_mtime for path in racecard_files)

    # --- Load model ---
    model_path = output_dir / "model.txt"
    if not model_path.exists():
        logger.error("Model not found at %s. Run --step train first.", model_path)
        return

    booster = lgb.Booster(model_file=str(model_path))

    with open(output_dir / "feature_cols.json") as f:
        feature_cols: list[str] = json.load(f)

    logger.info("Loaded model (%d iterations) and %d feature columns", booster.best_iteration, len(feature_cols))

    # --- Load features ---
    features_2026 = PROJECT_ROOT / config["paths"]["marts"] / "features_2026.parquet"
    if not features_2026.exists():
        logger.error(
            "features_2026.parquet not found. Refusing to fall back to historical validation data. "
            "Run --step fetch-racecards and --step features first."
        )
        raise RuntimeError("Missing features_2026.parquet")
    if features_2026.stat().st_mtime < latest_racecard_mtime:
        raise RuntimeError(
            "features_2026.parquet is older than the latest racecard JSON. "
            "Run --step features before --step predict."
        )
    df = pd.read_parquet(features_2026)
    if df.empty:
        logger.error(
            "features_2026.parquet is empty. Ensure Cheltenham 2026 racecards were fetched, then rerun --step features."
        )
        raise RuntimeError("features_2026.parquet is empty")
    logger.info("Loaded features_2026: %s", df.shape)

    if "going_bucket" in df.columns:
        missing_going = df["going_bucket"].isna()
        if missing_going.any():
            logger.warning(
                "Going is missing for %d/%d runners across %d races; rerun fetch-racecards before final bet decisions.",
                int(missing_going.sum()),
                len(df),
                int(df.loc[missing_going, "race_id"].nunique()),
            )

    # --- Score ---
    df["raw_prob"] = booster.predict(df[feature_cols], num_iteration=booster.best_iteration)
    df["win_prob"] = _softmax_per_race(df, "raw_prob")  # type: ignore[arg-type]

    # --- Harville place probabilities ---
    num_places: int = model_cfg["harville"]["num_places"]
    place_probs_list: list[pd.Series] = []  # type: ignore[type-arg]

    for _race_id, group in df.groupby("race_id"):
        wp = np.asarray(group["win_prob"], dtype=np.float64)
        pp = _harville_place_probs(wp, num_places=num_places)
        place_probs_list.append(pd.Series(pp, index=group.index))

    df["place_prob"] = pd.concat(place_probs_list)
    logger.info("Computed Harville place probabilities (top %d)", num_places)

    # --- Value overlay ---
    runners_path = PROJECT_ROOT / config["paths"]["staged_parquet"] / "runners.parquet"
    runners = pd.read_parquet(runners_path)
    sp = runners[["race_id", "horse_id", "sp_decimal"]].copy()
    df = df.merge(sp, on=["race_id", "horse_id"], how="left")

    # Prefer exchange back odds over SP when available
    exchange_path = PROJECT_ROOT / config["paths"]["raw_betfair"] / "exchange_odds.parquet"
    if exchange_path.exists():
        exchange = pd.read_parquet(exchange_path)
        if "wap" in exchange.columns:
            ex_odds: pd.DataFrame = exchange[["race_id", "horse_id", "wap"]].copy()  # type: ignore[assignment]
            ex_odds = ex_odds.rename(columns={"wap": "exchange_odds"})  # type: ignore[call-overload]
            df = df.merge(ex_odds, on=["race_id", "horse_id"], how="left")
            df["best_odds"] = df["exchange_odds"].fillna(df["sp_decimal"])
        else:
            df["best_odds"] = df["sp_decimal"]
    else:
        df["best_odds"] = df["sp_decimal"]

    df["implied_prob"] = np.where(
        df["best_odds"].notna() & (df["best_odds"] > 1),
        1.0 / df["best_odds"],
        np.nan,
    )
    df["value_score"] = df["win_prob"] - df["implied_prob"]

    strong_thresh: float = model_cfg["value"]["strong_threshold"]
    opp_thresh: float = model_cfg["value"]["opposable_threshold"]

    def _verdict(vs: float) -> str:
        if pd.isna(vs):
            return "No odds"
        if vs >= strong_thresh:
            return "Strong value"
        elif vs >= opp_thresh:
            return "Fair price"
        return "Opposable"

    df["verdict"] = df["value_score"].apply(_verdict)

    # --- Feature contributions ---
    logger.info("Computing feature contributions...")
    top_features_list: list[str] = []
    for idx in df.index:
        row = df.loc[[idx]]
        top_features_list.append(_top_features(booster, row, feature_cols))
    df["top_features"] = top_features_list

    # --- Output ---
    output_cols = [
        "race_id", "horse_id", "horse_name", "win_prob", "place_prob",
        "implied_prob", "value_score", "verdict", "top_features",
    ]

    # Include metadata if available for context
    extra_cols = ["date", "course", "race_name", "race_type", "race_class",
                  "off_time", "distance_f", "is_handicap", "pattern",
                  "going", "official_rating", "best_odds"]
    for extra in extra_cols:
        if extra in df.columns and extra not in output_cols:
            output_cols.append(extra)

    out_df = df[output_cols].copy()
    out_df = out_df.sort_values(by=["race_id", "win_prob"], ascending=[True, False])  # type: ignore[call-overload]
    plan_a_points_map = _build_plan_a_points(out_df, strong_thresh=strong_thresh)

    # Parquet
    out_df.to_parquet(output_dir / "predictions.parquet", index=False)
    logger.info("Saved predictions.parquet (%d rows)", len(out_df))

    # JSON grouped by race
    races_json: list[dict[str, Any]] = []
    race_meta_fields = ["date", "course", "race_name", "race_type", "race_class",
                        "off_time", "distance_f", "is_handicap", "pattern", "going"]
    for _race_id, group in out_df.groupby("race_id"):
        race_info: dict[str, Any] = {"race_id": str(_race_id)}
        for field in race_meta_fields:
            if field in group.columns:
                val = group[field].iloc[0]
                race_info[field] = None if pd.isna(val) else str(val)  # type: ignore[arg-type]
        race_info["analysis"] = _build_race_analysis(group, strong_thresh=strong_thresh)
        runners_list: list[dict[str, Any]] = []
        for row_idx in range(len(group)):
            r = group.iloc[row_idx]
            implied_raw = r["implied_prob"]
            value_raw = r["value_score"]
            implied = round(float(implied_raw), 4) if pd.notna(implied_raw) else None  # type: ignore[arg-type]
            value = round(float(value_raw), 4) if pd.notna(value_raw) else None  # type: ignore[arg-type]
            runner_dict: dict[str, Any] = {
                "horse_id": str(r["horse_id"]),
                "horse_name": str(r["horse_name"]),
                "win_prob": round(float(r["win_prob"]), 4),  # type: ignore[arg-type]
                "place_prob": round(float(r["place_prob"]), 4),  # type: ignore[arg-type]
                "implied_prob": implied,
                "value_score": value,
                "verdict": str(r["verdict"]),
                "top_features": str(r["top_features"]),
            }
            if "official_rating" in df.columns:
                or_val = r.get("official_rating")
                runner_dict["official_rating"] = int(or_val) if pd.notna(or_val) else None  # type: ignore[arg-type]
            if "best_odds" in df.columns:
                odds_val = r.get("best_odds")
                runner_dict["best_odds"] = round(float(odds_val), 2) if pd.notna(odds_val) else None  # type: ignore[arg-type]
            plan_key = (str(_race_id), str(r["horse_id"]))
            if plan_key in plan_a_points_map:
                runner_dict["plan_a_points"] = round(float(plan_a_points_map[plan_key]), 2)
            runners_list.append(runner_dict)
        plan_runners = [runner for runner in runners_list if "plan_a_points" in runner]
        if plan_runners:
            plan_runners = sorted(
                plan_runners,
                key=lambda item: float(item.get("plan_a_points", 0.0)),
                reverse=True,
            )
            race_info["plan_a_pick_horse_id"] = plan_runners[0]["horse_id"]
            race_info["plan_a_pick_horse_name"] = plan_runners[0]["horse_name"]
            race_info["plan_a_points"] = plan_runners[0]["plan_a_points"]
        race_info["runners"] = runners_list
        races_json.append(race_info)

    with open(output_dir / "predictions.json", "w") as f:
        json.dump(races_json, f, indent=2)

    logger.info("Saved predictions.json (%d races)", len(races_json))

    # Summary stats
    logger.info(
        "Win prob range: [%.4f, %.4f], Place prob range: [%.4f, %.4f]",
        float(out_df["win_prob"].min()), float(out_df["win_prob"].max()),  # type: ignore[arg-type]
        float(out_df["place_prob"].min()), float(out_df["place_prob"].max()),  # type: ignore[arg-type]
    )

    verdicts = out_df["verdict"].value_counts()
    logger.info("Verdict distribution:\n%s", verdicts.to_string())
