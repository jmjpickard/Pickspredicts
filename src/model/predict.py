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


def _build_race_analysis(group: pd.DataFrame, strong_thresh: float) -> str:
    """Build deterministic race-level narrative from model + market view."""
    if group.empty:
        return ""

    group = group.copy()
    group = group.sort_values("win_prob", ascending=False)
    model_top = group.iloc[0]
    model_name = str(model_top.get("horse_name", ""))
    model_wp = float(model_top.get("win_prob", 0.0))

    with_odds = group[
        group["best_odds"].notna()
        & (group["best_odds"] > 1)
        & group["value_score"].notna()
    ].copy()
    if with_odds.empty:
        return (
            f"{model_name} is top on model win chance ({_fmt_pct(model_wp)}), "
            "but there are no usable live odds yet."
        )

    market_fav = with_odds.sort_values("best_odds", ascending=True).iloc[0]
    value_top = with_odds.sort_values("value_score", ascending=False).iloc[0]

    market_name = str(market_fav.get("horse_name", ""))
    market_odds = float(market_fav.get("best_odds", np.nan))
    value_name = str(value_top.get("horse_name", ""))
    value_odds = float(value_top.get("best_odds", np.nan))
    value_score = float(value_top.get("value_score", 0.0))
    value_wp = float(value_top.get("win_prob", 0.0))
    implied = float(value_top.get("implied_prob", np.nan))

    positive_value = with_odds[with_odds["value_score"] > 0].copy()
    realistic_value = positive_value[positive_value["best_odds"] <= 30].copy()

    if value_score < 0:
        return (
            f"{model_name} leads on win chance ({_fmt_pct(model_wp)}), "
            f"but the market looks tight around {market_name} at {_fmt_odds(market_odds)}."
        )

    if value_score < 0.02 and realistic_value.empty and not positive_value.empty:
        outsider = positive_value.sort_values("value_score", ascending=False).iloc[0]
        outsider_name = str(outsider.get("horse_name", ""))
        return (
            "Front of market is tight/fair; clean value is mostly in speculative outsiders "
            f"such as {outsider_name}."
        )

    edge_label = "strong overlay" if value_score >= strong_thresh else "value edge"

    if value_name == model_name and value_name == market_name:
        return (
            f"{value_name} is both model top and market favourite, and still rates as a {edge_label} "
            f"({_fmt_pct(value_wp)} vs market {_fmt_pct(implied)})."
        )

    if value_name == model_name and value_name != market_name:
        return (
            f"{value_name} is model top and the best value angle at {_fmt_odds(value_odds)} "
            f"({_fmt_pct(value_wp)} vs market {_fmt_pct(implied)}), ahead of market favourite {market_name}."
        )

    if value_name != model_name:
        return (
            f"{model_name} leads on pure win chance ({_fmt_pct(model_wp)}), but {value_name} has the best value case "
            f"at {_fmt_odds(value_odds)} ({_fmt_pct(value_wp)} vs market {_fmt_pct(implied)})."
        )

    return (
        f"{model_name} leads at {_fmt_pct(model_wp)} and remains a fair value play at {_fmt_odds(value_odds)}."
    )


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
            runners_list.append(runner_dict)
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
