"""Compare recency-weighted V4 variants on the festival backtest windows.

Usage:
    .venv/bin/python scripts/compare_recency_weighting.py
    .venv/bin/python scripts/compare_recency_weighting.py --half-lives 180 365 730 --min-weights 0.2 0.35
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.backtest_festival_years as bf
from src.model.feature_groups import active_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WeightingSpec:
    key: str
    label: str
    sample_weights: dict[str, Any]


@dataclass
class WeightingReport:
    key: str
    label: str
    sample_weights: dict[str, Any]
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
    parser = argparse.ArgumentParser(description="Compare recency-weighted V4 variants")
    parser.add_argument(
        "--json-out",
        default="data/model/backtest/recency_weighting_comparison.json",
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
    parser.add_argument(
        "--half-lives",
        type=float,
        nargs="+",
        default=[180.0, 365.0, 730.0],
        help="Recency half-life candidates in days",
    )
    parser.add_argument(
        "--min-weights",
        type=float,
        nargs="+",
        default=[0.2, 0.35],
        help="Minimum weight floor candidates",
    )
    return parser.parse_args()


def _build_specs(base_sample_weights: dict[str, Any] | None, args: argparse.Namespace) -> list[WeightingSpec]:
    baseline_weights = copy.deepcopy(base_sample_weights) if base_sample_weights is not None else {"enabled": False}
    specs = [
        WeightingSpec(
            key="baseline",
            label="Baseline",
            sample_weights=baseline_weights,
        )
    ]
    for half_life in args.half_lives:
        for min_weight in args.min_weights:
            specs.append(
                WeightingSpec(
                    key=f"recency_{int(round(half_life))}d_floor_{str(min_weight).replace('.', '_')}",
                    label=f"Recency {half_life:g}d / floor {min_weight:.2f}",
                    sample_weights={
                        "enabled": True,
                        "strategy": "recency",
                        "recency": {
                            "half_life_days": float(half_life),
                            "min_weight": float(min_weight),
                            "normalize": True,
                        },
                    },
                )
            )
    return specs


def _summarise_report(spec: WeightingSpec, results: list[bf.BacktestResult]) -> WeightingReport:
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

    return WeightingReport(
        key=spec.key,
        label=spec.label,
        sample_weights=spec.sample_weights,
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


def _print_summary(reports: list[WeightingReport]) -> None:
    print("Recency Weighting Comparison (146 races across 6 festivals)\n")
    print(
        f"{'Variant':<28} {'LogLoss':>8} {'MRR':>7} {'Top1':>7} "
        f"{'Top1 ROI':>9} {'Strong ROI':>11} {'Comp':>8}"
    )
    for report in reports:
        strong_roi = 100 * report.strong_win_roi if report.strong_win_roi is not None else None
        print(
            f"{report.label:<28} {report.pooled_log_loss:>8.4f} {report.pooled_mrr:>7.4f} "
            f"{100 * report.top1_accuracy:>6.1f}% {100 * report.top1_roi:>8.1f}% "
            f"{(f'{strong_roi:.1f}%' if strong_roi is not None else 'NA'):>11} {report.composite:>8.4f}"
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
    feature_cols = active_feature_cols(all_feature_cols, model_cfg.get("feature_groups"))
    specs = _build_specs(model_cfg.get("sample_weights"), args)

    reports: list[WeightingReport] = []
    for spec in specs:
        results: list[bf.BacktestResult] = []
        for window_name in bf.DEFAULT_WINDOWS:
            course, start, end = bf.FESTIVAL_WINDOWS[window_name]
            result = bf.evaluate_window(
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
                sample_weights_cfg=spec.sample_weights,
            )
            results.append(result)
        reports.append(_summarise_report(spec, results))

    reports.sort(key=lambda report: report.composite)
    _print_summary(reports)

    out_path = PROJECT_ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(report) for report in reports], f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
