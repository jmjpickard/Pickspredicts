"""Bayesian hyperparameter search using Optuna TPE.

Replaces the one-at-a-time LLM mutation loop with proper multi-parameter
Bayesian optimization. Uses the same 6-window composite score as autoresearch.

Usage:
    .venv/bin/python scripts/optuna_search.py --trials 100
    .venv/bin/python scripts/optuna_search.py --trials 200 --data-mode v3
    .venv/bin/python scripts/optuna_search.py --trials 50 --apply
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from scripts.autoresearch import (
    META_COLS,
    evaluate_config,
    pipeline_baseline_config,
)
from src.model.feature_groups import (
    FEATURE_GROUPS,
    REQUIRED_FEATURE_GROUPS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

V3_COURSES = {"Cheltenham", "Aintree", "Punchestown", "Leopardstown"}

# Feature groups that should always be on (removing them always hurts)
ALWAYS_ON_GROUPS = {"race_context", "ratings", "horse_form", "connections", "runner_profile"}

# Feature groups that Optuna can toggle
TOGGLEABLE_GROUPS = [
    g for g in FEATURE_GROUPS
    if g not in ALWAYS_ON_GROUPS and g not in REQUIRED_FEATURE_GROUPS
]

WINDOWS = [
    "cheltenham_2023", "cheltenham_2024", "cheltenham_2025",
    "aintree_2023", "aintree_2024", "aintree_2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna Bayesian hyperparameter search")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument(
        "--data-mode",
        choices=["v3", "all"],
        default="v3",
        help="v3 = 4 festival courses only, all = 11 courses",
    )
    parser.add_argument("--study-name", default="cheltenham_v5", help="Optuna study name")
    parser.add_argument(
        "--out-dir",
        default="data/autoresearch/optuna",
        help="Output directory for results",
    )
    parser.add_argument("--apply", action="store_true", help="Write best config to pipeline.yaml")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for Optuna sampler")
    return parser.parse_args()


def load_data(
    data_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load features and runners, optionally filtering to V3 courses."""
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)

    features = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["marts"] / "features.parquet")
    runners = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "runners.parquet")
    features["date"] = pd.to_datetime(features["date"])

    if data_mode == "v3":
        features = features[features["course"].isin(V3_COURSES)].copy()

    all_feature_cols = [c for c in features.columns if c not in META_COLS]
    return features, runners, all_feature_cols


def build_config_from_trial(trial: optuna.Trial, baseline: dict[str, Any]) -> dict[str, Any]:
    """Construct a full config dict from an Optuna trial's suggested params."""
    cfg = copy.deepcopy(baseline)

    # LightGBM hyperparameters
    cfg["lgbm"]["num_leaves"] = trial.suggest_int("num_leaves", 16, 80)
    cfg["lgbm"]["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.10, log=True)
    cfg["lgbm"]["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
    cfg["lgbm"]["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
    cfg["lgbm"]["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
    cfg["lgbm"]["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True)
    cfg["lgbm"]["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-4, 5.0, log=True)
    cfg["lgbm"]["path_smooth"] = trial.suggest_float("path_smooth", 0.0, 15.0)

    # Feature group toggles
    for group in TOGGLEABLE_GROUPS:
        cfg["feature_groups"][group] = trial.suggest_categorical(
            f"fg_{group}", [True, False]
        )

    # Always-on groups
    for group in ALWAYS_ON_GROUPS:
        cfg["feature_groups"][group] = True

    return cfg


def create_objective(
    features: pd.DataFrame,
    runners: pd.DataFrame,
    all_feature_cols: list[str],
    baseline: dict[str, Any],
) -> optuna.study.ObjectiveFuncType:
    """Create an Optuna objective function that evaluates a trial config."""

    def objective(trial: optuna.Trial) -> float:
        cfg = build_config_from_trial(trial, baseline)
        result = evaluate_config(cfg, features, runners, all_feature_cols, WINDOWS)
        score = result["score"]

        # Store extra metrics as trial user attrs for analysis
        trial.set_user_attr("pooled_log_loss", result["pooled_log_loss"])
        trial.set_user_attr("pooled_mrr", result["pooled_mrr"])
        trial.set_user_attr("n_features", result["n_features"])

        return score

    return objective


def trial_to_config(trial: optuna.trial.FrozenTrial, baseline: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the full config from a completed trial's params."""
    cfg = copy.deepcopy(baseline)

    cfg["lgbm"]["num_leaves"] = trial.params["num_leaves"]
    cfg["lgbm"]["learning_rate"] = trial.params["learning_rate"]
    cfg["lgbm"]["feature_fraction"] = trial.params["feature_fraction"]
    cfg["lgbm"]["bagging_fraction"] = trial.params["bagging_fraction"]
    cfg["lgbm"]["min_child_samples"] = trial.params["min_child_samples"]
    cfg["lgbm"]["lambda_l1"] = trial.params["lambda_l1"]
    cfg["lgbm"]["lambda_l2"] = trial.params["lambda_l2"]
    cfg["lgbm"]["path_smooth"] = trial.params["path_smooth"]

    for group in TOGGLEABLE_GROUPS:
        cfg["feature_groups"][group] = trial.params[f"fg_{group}"]

    for group in ALWAYS_ON_GROUPS:
        cfg["feature_groups"][group] = True

    return cfg


def apply_to_pipeline(best_config: dict[str, Any]) -> None:
    """Write the best config back to pipeline.yaml."""
    pipeline_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(pipeline_path) as f:
        pipeline = yaml.safe_load(f)

    for key in (
        "num_leaves", "learning_rate", "feature_fraction", "bagging_fraction",
        "min_child_samples", "lambda_l1", "lambda_l2", "path_smooth",
    ):
        pipeline["model"]["lgbm"][key] = best_config["lgbm"][key]

    pipeline["model"]["feature_groups"] = best_config["feature_groups"]

    with open(pipeline_path, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
    print(f"Applied best config to {pipeline_path}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Suppress LightGBM training logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Loading data (mode={args.data_mode})...")
    features, runners, all_feature_cols = load_data(args.data_mode)
    print(f"  {len(features)} rows, {len(all_feature_cols)} feature columns")

    # Build baseline from current pipeline config
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        pipeline_cfg = yaml.safe_load(f)
    baseline = pipeline_baseline_config(pipeline_cfg)

    # Evaluate baseline first
    print("\nEvaluating baseline...")
    baseline_result = evaluate_config(baseline, features, runners, all_feature_cols, WINDOWS)
    print(
        f"  Baseline score: {baseline_result['score']:.4f} "
        f"(LL={baseline_result['pooled_log_loss']:.4f}, "
        f"MRR={baseline_result['pooled_mrr']:.4f}, "
        f"features={baseline_result['n_features']})"
    )

    # Create study with TPE sampler
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
    )

    # Enqueue the baseline as the first trial so TPE has a known-good starting point
    baseline_params: dict[str, Any] = {
        "num_leaves": baseline["lgbm"]["num_leaves"],
        "learning_rate": baseline["lgbm"]["learning_rate"],
        "feature_fraction": baseline["lgbm"]["feature_fraction"],
        "bagging_fraction": baseline["lgbm"]["bagging_fraction"],
        "min_child_samples": baseline["lgbm"]["min_child_samples"],
        "lambda_l1": max(baseline["lgbm"]["lambda_l1"], 1e-4),
        "lambda_l2": max(baseline["lgbm"]["lambda_l2"], 1e-4),
        "path_smooth": baseline["lgbm"]["path_smooth"],
    }
    for group in TOGGLEABLE_GROUPS:
        baseline_params[f"fg_{group}"] = baseline["feature_groups"].get(group, True)
    study.enqueue_trial(baseline_params)

    # Run optimisation
    objective = create_objective(features, runners, all_feature_cols, baseline)

    print(f"\nStarting Optuna search ({args.trials} trials, TPE sampler)...")
    print(f"Toggleable groups: {TOGGLEABLE_GROUPS}")
    print(f"Always-on groups: {list(ALWAYS_ON_GROUPS)}\n")

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_trial
        n = len(study.trials)
        ll = trial.user_attrs.get("pooled_log_loss", float("nan"))
        mrr = trial.user_attrs.get("pooled_mrr", float("nan"))
        marker = " ***" if trial.number == best.number else ""
        print(
            f"  [{n:>3}/{args.trials}] trial={trial.number} "
            f"score={trial.value:.4f} LL={ll:.4f} MRR={mrr:.4f}"
            f"{marker}"
        )

    study.optimize(objective, n_trials=args.trials, callbacks=[progress_callback])

    # Results
    best = study.best_trial
    best_config = trial_to_config(best, baseline)

    print(f"\n{'='*60}")
    print("OPTUNA SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Best trial: #{best.number}")
    print(f"  Best score: {best.value:.4f} (baseline: {baseline_result['score']:.4f})")
    print(
        f"  LL={best.user_attrs['pooled_log_loss']:.4f} "
        f"MRR={best.user_attrs['pooled_mrr']:.4f} "
        f"features={best.user_attrs['n_features']}"
    )
    improvement = baseline_result["score"] - best.value
    print(f"  Improvement: {improvement:+.4f} ({100*improvement/abs(baseline_result['score']):.1f}%)")

    print(f"\n  Best hyperparameters:")
    for key in ("num_leaves", "learning_rate", "feature_fraction", "bagging_fraction",
                "min_child_samples", "lambda_l1", "lambda_l2", "path_smooth"):
        base_val = baseline["lgbm"][key]
        best_val = best_config["lgbm"][key]
        changed = " *" if base_val != best_val else ""
        print(f"    {key}: {best_val}{changed}")

    print(f"\n  Feature groups:")
    for group in sorted(best_config["feature_groups"]):
        enabled = best_config["feature_groups"][group]
        base_enabled = baseline["feature_groups"].get(group, True)
        changed = " *" if enabled != base_enabled else ""
        print(f"    {group}: {enabled}{changed}")

    # Save outputs
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best config
    best_out = {
        "score": best.value,
        "trial": best.number,
        "total_trials": len(study.trials),
        "data_mode": args.data_mode,
        "baseline_score": baseline_result["score"],
        "pooled_log_loss": best.user_attrs["pooled_log_loss"],
        "pooled_mrr": best.user_attrs["pooled_mrr"],
        "n_features": best.user_attrs["n_features"],
        "config": best_config,
    }
    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best_out, f, indent=2)

    # Full trial history
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "score": t.value,
            "pooled_log_loss": t.user_attrs.get("pooled_log_loss"),
            "pooled_mrr": t.user_attrs.get("pooled_mrr"),
            "n_features": t.user_attrs.get("n_features"),
            "params": t.params,
        })
    with open(out_dir / "trial_history.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    # Top 10 trials
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))[:10]
    print(f"\n  Top 10 trials:")
    for t in top_trials:
        ll = t.user_attrs.get("pooled_log_loss", float("nan"))
        mrr = t.user_attrs.get("pooled_mrr", float("nan"))
        print(f"    #{t.number}: score={t.value:.4f} LL={ll:.4f} MRR={mrr:.4f}")

    print(f"\n  Saved to {out_dir}/")

    if args.apply:
        apply_to_pipeline(best_config)


if __name__ == "__main__":
    main()
