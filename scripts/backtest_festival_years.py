"""Walk-forward backtest for Cheltenham festival years.

Runs strict train-before-validation backtests for selected years, then reports:
- log loss
- top-pick accuracy / ROI
- favourite baseline
- "Strong value" win + each-way ROI
- bootstrap 95% confidence intervals for ROI estimates

Usage:
    .venv/bin/python scripts/backtest_festival_years.py
    .venv/bin/python scripts/backtest_festival_years.py --years 2023 2024 2025 --json-out data/model/backtest/festival_years.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

FESTIVAL_WINDOWS = {
    2023: ("2023-03-14", "2023-03-17"),
    2024: ("2024-03-12", "2024-03-15"),
    2025: ("2025-03-11", "2025-03-14"),
}


@dataclass
class BacktestResult:
    year: int
    train_rows: int
    val_rows: int
    val_races: int
    best_iteration: int
    log_loss: float
    top1_accuracy: float
    top1_roi: float
    top1_pnl: float
    top1_roi_ci_low: float | None
    top1_roi_ci_high: float | None
    fav_top1_accuracy: float
    fav_roi: float
    fav_pnl: float
    strong_bets: int
    strong_win_roi: float | None
    strong_win_pnl: float
    strong_win_roi_ci_low: float | None
    strong_win_roi_ci_high: float | None
    strong_ew_roi: float | None
    strong_ew_pnl: float


def load_config() -> dict[str, Any]:
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _softmax_per_race(df: pd.DataFrame, col: str = "raw_prob") -> pd.Series:
    return df.groupby("race_id")[col].transform(lambda x: x / x.sum())


def _bootstrap_ci(
    returns: np.ndarray[Any, np.dtype[np.floating[Any]]],
    n_bootstrap: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if len(returns) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns, size=(n_bootstrap, len(returns)), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _win_returns(df: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.where(df["won"] == 1, df["sp_decimal"] - 1, -1.0).astype(np.float64)


def _ew_returns(df: pd.DataFrame, races: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    race_terms = races[["race_id", "is_handicap", "field_size"]].drop_duplicates("race_id")
    race_terms = race_terms.rename(
        columns={"is_handicap": "race_is_handicap", "field_size": "race_field_size"}
    )
    merged = df.merge(race_terms, on="race_id", how="left")

    ew = np.zeros(len(merged), dtype=np.float64)
    for idx, (_, row) in enumerate(merged.iterrows()):
        sp = float(row["sp_decimal"])
        fp = int(row["finish_position"])
        is_handicap = bool(row["race_is_handicap"]) if pd.notna(row["race_is_handicap"]) else False
        field_size = int(row["race_field_size"]) if pd.notna(row["race_field_size"]) else 0
        if is_handicap and field_size >= 16:
            num_places = 4
            place_fraction = 1 / 5
        else:
            num_places = 3
            place_fraction = 1 / 4

        ew_win = 0.5 * (sp - 1) if fp == 1 else -0.5
        place_odds = 1 + (sp - 1) * place_fraction
        ew_place = 0.5 * (place_odds - 1) if fp <= num_places else -0.5
        ew[idx] = ew_win + ew_place
    return ew


def evaluate_year(
    year: int,
    start: str,
    end: str,
    course: str,
    strong_threshold: float,
    n_bootstrap: int,
    seed: int,
    features: pd.DataFrame,
    runners: pd.DataFrame,
    races: pd.DataFrame,
    feature_cols: list[str],
    lgbm_params: dict[str, Any],
) -> BacktestResult:
    val_start = pd.Timestamp(start)
    val_end = pd.Timestamp(end)

    train_df = features[features["date"] < val_start].copy()
    val_df = features[
        (features["course"] == course)
        & (features["date"] >= val_start)
        & (features["date"] <= val_end)
    ].copy()

    # Canonical SP/result fields from staged runners for settlement.
    val_df = val_df.drop(columns=["sp_decimal"], errors="ignore")
    val_df = val_df.merge(
        runners[["race_id", "horse_id", "sp_decimal", "finish_position"]],
        on=["race_id", "horse_id"],
        how="left",
        suffixes=("", "_runner"),
    )
    if "finish_position_runner" in val_df.columns:
        val_df["finish_position"] = val_df["finish_position_runner"]
        val_df = val_df.drop(columns=["finish_position_runner"])

    train_params = {k: v for k, v in lgbm_params.items() if k != "early_stopping_rounds"}

    train_ds = lgb.Dataset(train_df[feature_cols], label=train_df["won"])
    val_ds = lgb.Dataset(val_df[feature_cols], label=val_df["won"], reference=train_ds)

    booster = lgb.train(
        train_params,
        train_ds,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(lgbm_params["early_stopping_rounds"]),
            lgb.log_evaluation(period=0),
        ],
    )

    val_df["raw_prob"] = booster.predict(val_df[feature_cols], num_iteration=booster.best_iteration)
    val_df["win_prob"] = _softmax_per_race(val_df, "raw_prob")
    ll = float(log_loss(val_df["won"], val_df["win_prob"]))

    # Model top pick
    top_idx = val_df.groupby("race_id")["win_prob"].idxmax()
    top_picks = val_df.loc[top_idx].copy()
    top_returns = _win_returns(top_picks)
    top_ci = _bootstrap_ci(top_returns, n_bootstrap=n_bootstrap, seed=seed + year)

    top1_acc = float((top_picks["won"] == 1).mean()) if len(top_picks) else 0.0
    top1_roi = float(top_returns.mean()) if len(top_returns) else 0.0
    top1_pnl = float(top_returns.sum()) if len(top_returns) else 0.0

    # Favourite baseline
    fav_idx = val_df.groupby("race_id")["sp_decimal"].idxmin()
    fav_picks = val_df.loc[fav_idx].copy()
    fav_returns = _win_returns(fav_picks)
    fav_acc = float((fav_picks["won"] == 1).mean()) if len(fav_picks) else 0.0
    fav_roi = float(fav_returns.mean()) if len(fav_returns) else 0.0
    fav_pnl = float(fav_returns.sum()) if len(fav_returns) else 0.0

    # Strategy settlement ("Strong value")
    val_df["implied_prob"] = 1.0 / val_df["sp_decimal"]
    val_df["value_score"] = val_df["win_prob"] - val_df["implied_prob"]
    strong_picks = val_df[
        (val_df["value_score"] >= strong_threshold)
        & val_df["sp_decimal"].notna()
        & val_df["finish_position"].notna()
    ].copy()

    if strong_picks.empty:
        strong_win_roi = None
        strong_win_pnl = 0.0
        strong_win_ci = (None, None)
        strong_ew_roi = None
        strong_ew_pnl = 0.0
    else:
        strong_win = np.where(
            strong_picks["finish_position"] == 1,
            strong_picks["sp_decimal"] - 1,
            -1.0,
        ).astype(np.float64)
        strong_win_roi = float(strong_win.mean())
        strong_win_pnl = float(strong_win.sum())
        strong_win_ci = _bootstrap_ci(strong_win, n_bootstrap=n_bootstrap, seed=seed + 1000 + year)

        strong_ew = _ew_returns(strong_picks, races)
        strong_ew_roi = float(strong_ew.mean())
        strong_ew_pnl = float(strong_ew.sum())

    return BacktestResult(
        year=year,
        train_rows=len(train_df),
        val_rows=len(val_df),
        val_races=int(val_df["race_id"].nunique()),
        best_iteration=int(booster.best_iteration),
        log_loss=ll,
        top1_accuracy=top1_acc,
        top1_roi=top1_roi,
        top1_pnl=top1_pnl,
        top1_roi_ci_low=top_ci[0],
        top1_roi_ci_high=top_ci[1],
        fav_top1_accuracy=fav_acc,
        fav_roi=fav_roi,
        fav_pnl=fav_pnl,
        strong_bets=len(strong_picks),
        strong_win_roi=strong_win_roi,
        strong_win_pnl=strong_win_pnl,
        strong_win_roi_ci_low=strong_win_ci[0],
        strong_win_roi_ci_high=strong_win_ci[1],
        strong_ew_roi=strong_ew_roi,
        strong_ew_pnl=strong_ew_pnl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Festival walk-forward backtest")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Festival years to evaluate",
    )
    parser.add_argument("--course", default="Cheltenham", help="Validation course")
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=None,
        help="Override 'Strong value' threshold (default from config)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20000,
        help="Bootstrap samples for ROI confidence intervals",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed")
    parser.add_argument(
        "--json-out",
        default="data/model/backtest/festival_years.json",
        help="Path to JSON output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    model_cfg = cfg["model"]
    strong_threshold = (
        float(args.strong_threshold)
        if args.strong_threshold is not None
        else float(model_cfg["value"]["strong_threshold"])
    )

    features = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["marts"] / "features.parquet")
    runners = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "runners.parquet")
    races = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "races.parquet")

    features["date"] = pd.to_datetime(features["date"])
    feature_cols = [c for c in features.columns if c not in META_COLS]

    results: list[BacktestResult] = []
    for year in args.years:
        if year not in FESTIVAL_WINDOWS:
            raise ValueError(f"No configured festival window for year {year}")
        start, end = FESTIVAL_WINDOWS[year]
        result = evaluate_year(
            year=year,
            start=start,
            end=end,
            course=args.course,
            strong_threshold=strong_threshold,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed,
            features=features,
            runners=runners,
            races=races,
            feature_cols=feature_cols,
            lgbm_params=model_cfg["lgbm"],
        )
        results.append(result)

    print(
        "year  races  logloss  top1_acc  top1_roi  fav_roi  strong_bets  strong_win_roi  strong_ew_roi"
    )
    for row in results:
        strong_win = f"{row.strong_win_roi:+.3f}" if row.strong_win_roi is not None else "NA"
        strong_ew = f"{row.strong_ew_roi:+.3f}" if row.strong_ew_roi is not None else "NA"
        print(
            f"{row.year}  {row.val_races:>5}  {row.log_loss:>7.4f}  "
            f"{row.top1_accuracy:>8.3f}  {row.top1_roi:>8.3f}  {row.fav_roi:>7.3f}  "
            f"{row.strong_bets:>11}  {strong_win:>14}  {strong_ew:>13}"
        )

    out_path = PROJECT_ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
