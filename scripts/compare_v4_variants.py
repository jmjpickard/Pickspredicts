"""Compare V4-style market-light variants on the six festival backtest windows.

Variants:
- V4 current: expanded-data config, market disabled, sp_rank retained
- V4 no sp_rank: also disable the market_proxy group
- V4 ensemble: 5-seed average of the current V4 config

Usage:
    .venv/bin/python scripts/compare_v4_variants.py
    .venv/bin/python scripts/compare_v4_variants.py --json-out data/model/backtest/v4_variants.json
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import scripts.backtest_festival_years as bf
from src.model.feature_groups import active_feature_cols
from src.model.sample_weights import build_sample_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class VariantSpec:
    key: str
    label: str
    feature_groups: dict[str, bool]
    seeds: list[int]


@dataclass
class VariantReport:
    key: str
    label: str
    feature_count: int
    seeds: list[int]
    pooled_log_loss: float
    pooled_mrr: float
    composite: float
    top1_accuracy: float
    top1_roi: float
    top1_pnl: float
    fav_top1_accuracy: float
    fav_roi: float
    fav_pnl: float
    strong_bets: int
    strong_win_roi: float | None
    strong_win_pnl: float
    strong_ew_roi: float | None
    strong_ew_pnl: float
    windows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V4 market-light variants")
    parser.add_argument(
        "--json-out",
        default="data/model/backtest/v4_variant_comparison.json",
        help="Destination JSON path for detailed results",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for ROI confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for bootstrap sampling",
    )
    return parser.parse_args()


def _params_for_seed(lgbm_params: dict[str, Any], seed: int) -> dict[str, Any]:
    params = {k: v for k, v in lgbm_params.items() if k != "early_stopping_rounds"}
    params["seed"] = seed
    return params


def evaluate_window_multi_seed(
    window_name: str,
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
    seeds: list[int],
    sample_weights_cfg: dict[str, Any] | None = None,
) -> bf.BacktestResult:
    val_start = pd.Timestamp(start)
    val_end = pd.Timestamp(end)

    train_df = features[features["date"] < val_start].copy()
    val_df = features[
        (features["course"] == course)
        & (features["date"] >= val_start)
        & (features["date"] <= val_end)
    ].copy()

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

    pred_matrix: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    best_iterations: list[int] = []
    train_weights = build_sample_weights(train_df, val_start, sample_weights_cfg)

    for model_seed in seeds:
        train_ds = lgb.Dataset(
            train_df[feature_cols],
            label=train_df["won"],
            weight=train_weights.to_numpy(dtype=np.float64) if train_weights is not None else None,
        )
        val_ds = lgb.Dataset(val_df[feature_cols], label=val_df["won"], reference=train_ds)
        booster = lgb.train(
            _params_for_seed(lgbm_params, model_seed),
            train_ds,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(lgbm_params["early_stopping_rounds"]),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_matrix.append(
            np.asarray(
                booster.predict(val_df[feature_cols], num_iteration=booster.best_iteration),
                dtype=np.float64,
            )
        )
        best_iterations.append(int(booster.best_iteration))

    val_df = val_df.copy()
    val_df["raw_prob"] = np.mean(np.vstack(pred_matrix), axis=0)
    val_df["win_prob"] = bf._softmax_per_race(val_df, "raw_prob")
    ll = float(bf.log_loss(val_df["won"], val_df["win_prob"]))
    mrr = bf._mean_reciprocal_rank(val_df)

    top_idx = val_df.groupby("race_id")["win_prob"].idxmax()
    top_picks = val_df.loc[top_idx].copy()
    top_returns = bf._win_returns(top_picks)
    window_seed = abs(hash(window_name)) % 100000
    top_ci = bf._bootstrap_ci(top_returns, n_bootstrap=n_bootstrap, seed=seed + window_seed)

    top1_acc = float((top_picks["won"] == 1).mean()) if len(top_picks) else 0.0
    top1_roi = float(top_returns.mean()) if len(top_returns) else 0.0
    top1_pnl = float(top_returns.sum()) if len(top_returns) else 0.0

    fav_idx = val_df.groupby("race_id")["sp_decimal"].idxmin()
    fav_picks = val_df.loc[fav_idx].copy()
    fav_returns = bf._win_returns(fav_picks)
    fav_acc = float((fav_picks["won"] == 1).mean()) if len(fav_picks) else 0.0
    fav_roi = float(fav_returns.mean()) if len(fav_returns) else 0.0
    fav_pnl = float(fav_returns.sum()) if len(fav_returns) else 0.0

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
        strong_win_ci = bf._bootstrap_ci(
            strong_win, n_bootstrap=n_bootstrap, seed=seed + 1000 + window_seed
        )

        strong_ew = bf._ew_returns(strong_picks, races)
        strong_ew_roi = float(strong_ew.mean())
        strong_ew_pnl = float(strong_ew.sum())

    return bf.BacktestResult(
        window=window_name,
        course=course,
        train_rows=len(train_df),
        val_rows=len(val_df),
        val_races=int(val_df["race_id"].nunique()),
        best_iteration=int(round(float(np.mean(best_iterations)))) if best_iterations else 0,
        log_loss=ll,
        mrr=mrr,
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


def _summarise_variant(
    spec: VariantSpec,
    results: list[bf.BacktestResult],
    feature_count: int,
) -> VariantReport:
    total_races = sum(result.val_races for result in results)
    pooled_log_loss = float(np.mean([result.log_loss for result in results]))
    pooled_mrr = float(np.mean([result.mrr for result in results]))
    composite = pooled_log_loss - (0.5 * pooled_mrr)
    top1_pnl = float(sum(result.top1_pnl for result in results))
    fav_pnl = float(sum(result.fav_pnl for result in results))
    strong_bets = int(sum(result.strong_bets for result in results))
    strong_win_pnl = float(sum(result.strong_win_pnl for result in results))
    strong_ew_pnl = float(sum(result.strong_ew_pnl for result in results))

    top1_wins = int(round(sum(result.top1_accuracy * result.val_races for result in results)))
    fav_wins = int(round(sum(result.fav_top1_accuracy * result.val_races for result in results)))

    return VariantReport(
        key=spec.key,
        label=spec.label,
        feature_count=feature_count,
        seeds=spec.seeds,
        pooled_log_loss=pooled_log_loss,
        pooled_mrr=pooled_mrr,
        composite=composite,
        top1_accuracy=(top1_wins / total_races) if total_races else 0.0,
        top1_roi=(top1_pnl / total_races) if total_races else 0.0,
        top1_pnl=top1_pnl,
        fav_top1_accuracy=(fav_wins / total_races) if total_races else 0.0,
        fav_roi=(fav_pnl / total_races) if total_races else 0.0,
        fav_pnl=fav_pnl,
        strong_bets=strong_bets,
        strong_win_roi=(strong_win_pnl / strong_bets) if strong_bets else None,
        strong_win_pnl=strong_win_pnl,
        strong_ew_roi=(strong_ew_pnl / strong_bets) if strong_bets else None,
        strong_ew_pnl=strong_ew_pnl,
        windows=[asdict(result) for result in results],
    )


def _print_summary(reports: list[VariantReport]) -> None:
    print("Pooled Results (146 races across 6 festivals)\n")
    print(
        f"{'Variant':<24} {'Feats':>5} {'Seeds':>5} {'LogLoss':>8} {'MRR':>7} "
        f"{'Top1':>7} {'Top1 ROI':>9} {'Top1 P&L':>10} {'Comp':>8}"
    )
    for report in reports:
        print(
            f"{report.label:<24} {report.feature_count:>5} {len(report.seeds):>5} "
            f"{report.pooled_log_loss:>8.4f} {report.pooled_mrr:>7.4f} "
            f"{100 * report.top1_accuracy:>6.1f}% {100 * report.top1_roi:>8.1f}% "
            f"{report.top1_pnl:>+9.1f} {report.composite:>8.4f}"
        )

    baseline = reports[0]
    print(
        f"{'Mkt Favourite':<24} {'—':>5} {'—':>5} {'—':>8} {'—':>7} "
        f"{100 * baseline.fav_top1_accuracy:>6.1f}% {100 * baseline.fav_roi:>8.1f}% "
        f"{baseline.fav_pnl:>+9.1f} {'—':>8}"
    )

    print("\nStrong Value Betting\n")
    print(
        f"{'Variant':<24} {'Bets':>5} {'Win P&L':>10} {'Win ROI':>9} {'E/W P&L':>10} {'E/W ROI':>9}"
    )
    for report in reports:
        strong_win_roi = 100 * report.strong_win_roi if report.strong_win_roi is not None else None
        strong_ew_roi = 100 * report.strong_ew_roi if report.strong_ew_roi is not None else None
        print(
            f"{report.label:<24} {report.strong_bets:>5} {report.strong_win_pnl:>+9.1f} "
            f"{(f'{strong_win_roi:.1f}%' if strong_win_roi is not None else 'NA'):>9} "
            f"{report.strong_ew_pnl:>+9.1f} "
            f"{(f'{strong_ew_roi:.1f}%' if strong_ew_roi is not None else 'NA'):>9}"
        )


def main() -> None:
    args = parse_args()
    cfg = bf.load_config()
    model_cfg = cfg["model"]
    strong_threshold = float(model_cfg["value"]["strong_threshold"])

    features = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["marts"] / "features.parquet")
    runners = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "runners.parquet")
    races = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "races.parquet")
    features["date"] = pd.to_datetime(features["date"])
    all_feature_cols = [column for column in features.columns if column not in bf.META_COLS]

    base_groups = copy.deepcopy(model_cfg.get("feature_groups", {}))
    v4_no_sp_rank = copy.deepcopy(base_groups)
    v4_no_sp_rank["market_proxy"] = False

    variants = [
        VariantSpec(
            key="v4_current",
            label="V4 current",
            feature_groups=base_groups,
            seeds=[42],
        ),
        VariantSpec(
            key="v4_no_sp_rank",
            label="V4 no sp_rank",
            feature_groups=v4_no_sp_rank,
            seeds=[42],
        ),
        VariantSpec(
            key="v4_ensemble_5",
            label="V4 ensemble (5)",
            feature_groups=base_groups,
            seeds=[42, 43, 44, 45, 46],
        ),
    ]

    reports: list[VariantReport] = []
    for spec in variants:
        feature_cols = active_feature_cols(all_feature_cols, spec.feature_groups)
        results: list[bf.BacktestResult] = []
        for window_name in bf.DEFAULT_WINDOWS:
            course, start, end = bf.FESTIVAL_WINDOWS[window_name]
            result = evaluate_window_multi_seed(
                window_name=window_name,
                start=start,
                end=end,
                course=course,
                strong_threshold=strong_threshold,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
                features=features,
                runners=runners,
                races=races,
                feature_cols=feature_cols,
                lgbm_params=model_cfg["lgbm"],
                seeds=spec.seeds,
                sample_weights_cfg=model_cfg.get("sample_weights"),
            )
            results.append(result)
        reports.append(_summarise_variant(spec, results, len(feature_cols)))

    _print_summary(reports)

    out_path = PROJECT_ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(report) for report in reports], f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
