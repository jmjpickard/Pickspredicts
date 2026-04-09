"""Isolation test: V3 data (4 courses) with market=true vs market=false.

V3→V4 changed two things at once (expanded data + market=false).
This script isolates the market effect on V3's original 4-course data.

Usage:
    .venv/bin/python scripts/isolate_v3_market.py
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.backtest_festival_years as bf
from src.model.feature_groups import active_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V3_COURSES = {"Cheltenham", "Aintree", "Punchestown", "Leopardstown"}


def evaluate_variant(
    label: str,
    features: pd.DataFrame,
    runners: pd.DataFrame,
    races: pd.DataFrame,
    all_feature_cols: list[str],
    lgbm_params: dict[str, Any],
    feature_groups: dict[str, bool],
    strong_threshold: float,
) -> dict[str, Any]:
    feature_cols = active_feature_cols(all_feature_cols, feature_groups)
    results: list[bf.BacktestResult] = []

    for window_name in bf.DEFAULT_WINDOWS:
        course, start, end = bf.FESTIVAL_WINDOWS[window_name]
        result = bf.evaluate_window(
            window_name=window_name,
            start=start,
            end=end,
            course=course,
            strong_threshold=strong_threshold,
            n_bootstrap=2000,
            seed=42,
            features=features,
            runners=runners,
            races=races,
            feature_cols=feature_cols,
            lgbm_params=lgbm_params,
        )
        results.append(result)

    pooled_ll = float(np.mean([r.log_loss for r in results]))
    pooled_mrr = float(np.mean([r.mrr for r in results]))
    composite = pooled_ll - 0.5 * pooled_mrr
    total_races = sum(r.val_races for r in results)
    top1_pnl = sum(r.top1_pnl for r in results)
    strong_bets = sum(r.strong_bets for r in results)
    strong_win_pnl = sum(r.strong_win_pnl for r in results)
    strong_ew_pnl = sum(r.strong_ew_pnl for r in results)
    top1_wins = sum(round(r.top1_accuracy * r.val_races) for r in results)

    return {
        "label": label,
        "feature_count": len(feature_cols),
        "feature_groups": feature_groups,
        "pooled_log_loss": pooled_ll,
        "pooled_mrr": pooled_mrr,
        "composite": composite,
        "top1_accuracy": top1_wins / total_races if total_races else 0.0,
        "top1_roi": top1_pnl / total_races if total_races else 0.0,
        "top1_pnl": top1_pnl,
        "strong_bets": strong_bets,
        "strong_win_roi": strong_win_pnl / strong_bets if strong_bets else None,
        "strong_win_pnl": strong_win_pnl,
        "strong_ew_roi": strong_ew_pnl / strong_bets if strong_bets else None,
        "strong_ew_pnl": strong_ew_pnl,
        "windows": [asdict(r) for r in results],
    }


def main() -> None:
    cfg = bf.load_config()
    model_cfg = cfg["model"]
    lgbm_params = model_cfg["lgbm"]
    strong_threshold = float(model_cfg["value"]["strong_threshold"])

    # Load full data then filter to V3's 4 courses
    features = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["marts"] / "features.parquet")
    runners = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "runners.parquet")
    races = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "races.parquet")

    features["date"] = pd.to_datetime(features["date"])
    features_v3 = features[features["course"].isin(V3_COURSES)].copy()

    print(f"Full data: {len(features)} rows, V3 data (4 courses): {len(features_v3)} rows")

    all_feature_cols = [c for c in features.columns if c not in bf.META_COLS]

    # V3 config: all groups on (84 features)
    v3_market_on = {
        "race_context": True, "ratings": True, "horse_form": True,
        "connections": True, "pedigree": True, "market_proxy": True,
        "runner_profile": True, "market": True, "comments": True,
        "ratings_vs_field": True, "enhanced": True,
        "connections_extended": True, "horse_context": True,
    }

    # V3 data + market=false (isolate market removal)
    v3_market_off = copy.deepcopy(v3_market_on)
    v3_market_off["market"] = False

    # V3 data + market=false + market_proxy=false (truly market-free)
    v3_no_market = copy.deepcopy(v3_market_off)
    v3_no_market["market_proxy"] = False

    variants = [
        ("V3 market=on (baseline)", v3_market_on),
        ("V3 market=off", v3_market_off),
        ("V3 fully market-free", v3_no_market),
    ]

    reports = []
    for label, fg in variants:
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")
        report = evaluate_variant(
            label=label,
            features=features_v3,
            runners=runners,
            races=races,
            all_feature_cols=all_feature_cols,
            lgbm_params=lgbm_params,
            feature_groups=fg,
            strong_threshold=strong_threshold,
        )
        reports.append(report)
        print(
            f"  Composite: {report['composite']:.4f}  "
            f"LL: {report['pooled_log_loss']:.4f}  "
            f"MRR: {report['pooled_mrr']:.4f}  "
            f"Top1 ROI: {100*report['top1_roi']:.1f}%  "
            f"Strong Win ROI: {100*report['strong_win_roi']:.1f}%"
            if report["strong_win_roi"] is not None
            else f"  No strong bets"
        )

    # Summary table
    print(f"\n{'='*60}")
    print("ISOLATION TEST RESULTS (V3 data, 4 courses)")
    print(f"{'='*60}\n")
    print(
        f"{'Variant':<26} {'Feats':>5} {'LogLoss':>8} {'MRR':>7} "
        f"{'Top1':>7} {'Top1 ROI':>9} {'Comp':>8} {'SV Bets':>7} {'SV Win ROI':>10}"
    )
    for r in reports:
        sv_roi = f"{100*r['strong_win_roi']:.1f}%" if r["strong_win_roi"] is not None else "N/A"
        print(
            f"{r['label']:<26} {r['feature_count']:>5} "
            f"{r['pooled_log_loss']:>8.4f} {r['pooled_mrr']:>7.4f} "
            f"{100*r['top1_accuracy']:>6.1f}% {100*r['top1_roi']:>8.1f}% "
            f"{r['composite']:>8.4f} {r['strong_bets']:>7} {sv_roi:>10}"
        )

    out_path = PROJECT_ROOT / "data/model/backtest/v3_market_isolation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
