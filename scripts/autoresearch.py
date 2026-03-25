"""Autoresearch loop — Karpathy-style iterative model improvement.

Mutate one thing, backtest on fixed festival windows, keep if better, repeat.

Primary metric : pooled log loss across all windows (lower = better)
Secondary metric: pooled MRR — mean reciprocal rank of the winner (higher = better)

The "score" used to decide keep/discard is:
    score = pooled_log_loss - 0.5 * pooled_mrr   (lower = better)

This rewards both probability calibration AND ranking the winner highly.

Usage:
    .venv/bin/python scripts/autoresearch.py
    .venv/bin/python scripts/autoresearch.py --iterations 200 --seed 7
    .venv/bin/python scripts/autoresearch.py --windows cheltenham_2023 cheltenham_2024 cheltenham_2025
    .venv/bin/python scripts/autoresearch.py --iterations 50 --out data/autoresearch/run_test

Each iteration appends one line to <out>/results.jsonl.
Best config is always written to <out>/best_config.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Festival windows — identical to backtest_festival_years.py
# ---------------------------------------------------------------------------

FESTIVAL_WINDOWS: dict[str, tuple[str, str, str]] = {
    "cheltenham_2023": ("Cheltenham", "2023-03-14", "2023-03-17"),
    "cheltenham_2024": ("Cheltenham", "2024-03-12", "2024-03-15"),
    "cheltenham_2025": ("Cheltenham", "2025-03-11", "2025-03-14"),
    "aintree_2023":    ("Aintree",     "2023-04-13", "2023-04-15"),
    "aintree_2024":    ("Aintree",     "2024-04-11", "2024-04-13"),
    "aintree_2025":    ("Aintree",     "2025-04-03", "2025-04-05"),
}

META_COLS = {"race_id", "horse_id", "date", "course", "horse_name", "finish_position", "won", "placed"}

# ---------------------------------------------------------------------------
# Feature groups — used for group-level masking experiments
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "race_context": [
        "field_size", "is_handicap", "race_type_encoded", "race_class_num",
        "is_grade1", "is_grade2", "is_grade3", "track_direction_encoded",
        "distance_band", "going_bucket",
    ],
    "ratings": [
        "or_current", "rpr_current", "ts_current",
        "or_best_last3", "or_best_last5", "rpr_best_last3", "rpr_best_last5",
        "or_rpr_diff", "or_trend_last5",
    ],
    "horse_form": [
        "age_at_race", "days_since_last_run", "career_runs", "career_wins", "career_places",
        "chase_starts", "hurdle_starts", "dnf_rate_last5", "avg_btn_last3", "btn_trend_last5",
        "headgear_changed", "first_time_headgear", "win_rate_overall", "place_rate_overall",
        "runs_last_90d", "runs_last_365d", "win_rate_going_bucket", "win_rate_dist_band",
        "win_rate_course", "win_rate_track_direction", "place_rate_track_direction",
    ],
    "connections": [
        "trainer_winpct_14d", "trainer_winpct_30d", "trainer_winpct_90d",
        "jockey_winpct_14d", "jockey_winpct_30d", "jockey_winpct_90d",
        "trainer_festival_winpct", "jockey_festival_winpct", "combo_winpct",
    ],
    "pedigree": [
        "sire_cheltenham_winpct", "sire_going_winpct", "sire_dist_winpct",
    ],
    "runner_profile": [
        "sp_rank", "weight_carried", "weight_vs_field_avg",
        "course_dist_winner", "days_since_last_win", "class_change",
    ],
    "market": [
        "market_implied_prob", "market_rank", "pre_price_move", "market_confidence",
    ],
    "comments": [
        "dominant_style_code", "pct_trouble", "pct_jumping_issues",
    ],
}

# Groups that are always kept (removing them collapses the model entirely)
REQUIRED_GROUPS = {"race_context"}

# ---------------------------------------------------------------------------
# Default config — mirrors pipeline.yaml baseline
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # LightGBM hyperparameters
    "lgbm": {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "num_iterations": 2000,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "seed": 42,
    },
    # Feature groups to include (all on by default)
    "feature_groups": {g: True for g in FEATURE_GROUPS},
}

# ---------------------------------------------------------------------------
# Mutation space
# ---------------------------------------------------------------------------

# Each entry: param_path (dot-separated key into config), type, and range/options
# Types: "int", "log_float", "float", "bool", "choice"
MUTATIONS: list[dict[str, Any]] = [
    # LightGBM hyperparams
    {"path": "lgbm.num_leaves",        "type": "int",       "low": 16,   "high": 96},
    {"path": "lgbm.learning_rate",     "type": "log_float", "low": 0.01, "high": 0.15},
    {"path": "lgbm.feature_fraction",  "type": "float",     "low": 0.5,  "high": 1.0},
    {"path": "lgbm.bagging_fraction",  "type": "float",     "low": 0.5,  "high": 1.0},
    {"path": "lgbm.min_child_samples", "type": "int",       "low": 5,    "high": 50},
    # Feature group toggles (excluding required groups)
    *[
        {"path": f"feature_groups.{g}", "type": "bool"}
        for g in FEATURE_GROUPS
        if g not in REQUIRED_GROUPS
    ],
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get(cfg: dict[str, Any], path: str) -> Any:
    keys = path.split(".")
    node: Any = cfg
    for k in keys:
        node = node[k]
    return node


def _set(cfg: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node: Any = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _active_feature_cols(all_feature_cols: list[str], cfg: dict[str, Any]) -> list[str]:
    """Return feature columns that are active under the current config."""
    disabled: set[str] = set()
    for group_name, cols in FEATURE_GROUPS.items():
        if not cfg["feature_groups"].get(group_name, True):
            disabled.update(cols)
    return [c for c in all_feature_cols if c not in disabled]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _softmax_per_race(df: pd.DataFrame, col: str = "raw_prob") -> pd.Series:  # type: ignore[type-arg]
    return df.groupby("race_id")[col].transform(lambda x: x / x.sum())


def _mrr(val_df: pd.DataFrame) -> float:
    """Mean reciprocal rank: 1/rank_of_winner averaged over races."""
    vals: list[float] = []
    for _, race in val_df.groupby("race_id"):
        ranked = race.sort_values("win_prob", ascending=False).reset_index(drop=True)
        winners = ranked[ranked["won"] == 1]
        if winners.empty:
            continue
        vals.append(1.0 / (int(winners.index[0]) + 1))
    return float(np.mean(vals)) if vals else 0.0


def _eval_window(
    window_name: str,
    features: pd.DataFrame,
    runners: pd.DataFrame,
    feature_cols: list[str],
    lgbm_params: dict[str, Any],
) -> dict[str, float]:
    """Train + evaluate one festival window. Returns log_loss and mrr."""
    course, start, end = FESTIVAL_WINDOWS[window_name]
    val_start = pd.Timestamp(start)
    val_end = pd.Timestamp(end)

    train_df = features[features["date"] < val_start].copy()
    val_df = features[
        (features["course"] == course)
        & (features["date"] >= val_start)
        & (features["date"] <= val_end)
    ].copy()

    if val_df.empty:
        logger.warning("No validation data for window %s — skipping", window_name)
        return {"log_loss": float("inf"), "mrr": 0.0, "val_races": 0}

    # Attach canonical SP from staged runners (needed for sp_decimal)
    val_df = val_df.drop(columns=["sp_decimal"], errors="ignore")
    val_df = val_df.merge(
        runners[["race_id", "horse_id", "sp_decimal"]],
        on=["race_id", "horse_id"],
        how="left",
    )

    # Guard: need at least 2 columns and rows in train
    active_cols = [c for c in feature_cols if c in train_df.columns and c in val_df.columns]
    if not active_cols or len(train_df) < 100:
        return {"log_loss": float("inf"), "mrr": 0.0, "val_races": 0}

    train_params = {k: v for k, v in lgbm_params.items() if k != "early_stopping_rounds"}
    early_stop = lgbm_params.get("early_stopping_rounds", 50)

    train_ds = lgb.Dataset(train_df[active_cols], label=train_df["won"])
    val_ds = lgb.Dataset(val_df[active_cols], label=val_df["won"], reference=train_ds)

    booster = lgb.train(
        train_params,
        train_ds,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_df = val_df.copy()
    val_df["raw_prob"] = booster.predict(val_df[active_cols], num_iteration=booster.best_iteration)
    val_df["win_prob"] = _softmax_per_race(val_df, "raw_prob")

    ll = float(log_loss(val_df["won"], val_df["win_prob"]))
    mrr_val = _mrr(val_df)

    return {"log_loss": ll, "mrr": mrr_val, "val_races": int(val_df["race_id"].nunique())}


def evaluate_config(
    cfg: dict[str, Any],
    features: pd.DataFrame,
    runners: pd.DataFrame,
    all_feature_cols: list[str],
    windows: list[str],
) -> dict[str, Any]:
    """Evaluate config across all windows. Returns pooled metrics."""
    feature_cols = _active_feature_cols(all_feature_cols, cfg)
    if len(feature_cols) == 0:
        return {"score": float("inf"), "pooled_log_loss": float("inf"), "pooled_mrr": 0.0, "windows": {}}

    window_results: dict[str, dict[str, float]] = {}
    for w in windows:
        window_results[w] = _eval_window(w, features, runners, feature_cols, cfg["lgbm"])

    valid = [r for r in window_results.values() if r["val_races"] > 0]
    if not valid:
        return {"score": float("inf"), "pooled_log_loss": float("inf"), "pooled_mrr": 0.0, "windows": window_results}

    pooled_ll = float(np.mean([r["log_loss"] for r in valid]))
    pooled_mrr = float(np.mean([r["mrr"] for r in valid]))
    # Score: lower is better. Subtract MRR contribution to reward winner-ranking.
    score = pooled_ll - 0.5 * pooled_mrr

    return {
        "score": score,
        "pooled_log_loss": pooled_ll,
        "pooled_mrr": pooled_mrr,
        "n_features": len(feature_cols),
        "windows": window_results,
    }


# ---------------------------------------------------------------------------
# Mutation sampler
# ---------------------------------------------------------------------------

def sample_mutation(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], str]:
    """Return a mutated copy of cfg and a description of the change."""
    new_cfg = copy.deepcopy(cfg)
    mutation = rng.choice(MUTATIONS)  # type: ignore[arg-type]
    path: str = mutation["path"]
    mtype: str = mutation["type"]

    current = _get(new_cfg, path)

    if mtype == "int":
        low, high = int(mutation["low"]), int(mutation["high"])
        new_val = int(rng.integers(low, high + 1))
    elif mtype == "float":
        low, high = float(mutation["low"]), float(mutation["high"])
        new_val = float(rng.uniform(low, high))
    elif mtype == "log_float":
        low, high = float(mutation["low"]), float(mutation["high"])
        new_val = float(math.exp(rng.uniform(math.log(low), math.log(high))))
    elif mtype == "bool":
        new_val = not current
    else:
        raise ValueError(f"Unknown mutation type: {mtype}")

    _set(new_cfg, path, new_val)
    desc = f"{path}: {current!r} → {new_val!r}"
    return new_cfg, desc


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    mutation: str
    score: float
    pooled_log_loss: float
    pooled_mrr: float
    n_features: int
    accepted: bool
    is_best: bool
    elapsed_s: float
    config: dict[str, Any]
    window_results: dict[str, Any]


def run_autoresearch(
    windows: list[str],
    n_iterations: int,
    seed: int,
    out_dir: Path,
    resume: bool,
) -> None:
    cfg = load_config()
    model_cfg = cfg["model"]

    features_path = PROJECT_ROOT / cfg["paths"]["marts"] / "features.parquet"
    runners_path = PROJECT_ROOT / cfg["paths"]["staged_parquet"] / "runners.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {features_path}\n"
            "Run: python -m src.pipeline --step features"
        )

    logger.info("Loading features from %s", features_path)
    features = pd.read_parquet(features_path)
    runners = pd.read_parquet(runners_path)
    features["date"] = pd.to_datetime(features["date"])

    all_feature_cols = [c for c in features.columns if c not in META_COLS]
    logger.info("Feature matrix: %d rows, %d feature cols", len(features), len(all_feature_cols))

    # Validate windows — skip any with no data
    available_windows = []
    for w in windows:
        course, start, end = FESTIVAL_WINDOWS[w]
        wdf = features[
            (features["course"] == course)
            & (features["date"] >= pd.Timestamp(start))
            & (features["date"] <= pd.Timestamp(end))
        ]
        if wdf.empty:
            logger.warning("Window %s has no data in features.parquet — skipping", w)
        else:
            available_windows.append(w)
            logger.info("  Window %-22s  %d runners, %d races", w, len(wdf), wdf["race_id"].nunique())

    if not available_windows:
        raise RuntimeError("No valid windows found — check that features.parquet contains the expected courses/dates")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    best_config_path = out_dir / "best_config.json"

    # Resume from previous best if requested
    if resume and best_config_path.exists():
        with open(best_config_path) as f:
            saved = json.load(f)
        best_cfg = saved["config"]
        best_score = saved["score"]
        start_iter = saved.get("iteration", 0) + 1
        logger.info("Resuming from iteration %d, best score=%.4f", start_iter, best_score)
    else:
        best_cfg = copy.deepcopy(DEFAULT_CONFIG)
        # Score the baseline first
        logger.info("Evaluating baseline config...")
        baseline = evaluate_config(best_cfg, features, runners, all_feature_cols, available_windows)
        best_score = baseline["score"]
        start_iter = 0
        logger.info(
            "Baseline: score=%.4f  log_loss=%.4f  mrr=%.4f  features=%d",
            best_score, baseline["pooled_log_loss"], baseline["pooled_mrr"], baseline["n_features"],
        )
        _save_best(best_config_path, best_cfg, best_score, 0, baseline)

    rng = np.random.default_rng(seed)

    for i in range(start_iter, start_iter + n_iterations):
        t0 = time.time()
        candidate_cfg, mutation_desc = sample_mutation(best_cfg, rng)

        metrics = evaluate_config(candidate_cfg, features, runners, all_feature_cols, available_windows)
        elapsed = time.time() - t0

        accepted = metrics["score"] < best_score
        is_best = accepted

        if accepted:
            best_cfg = candidate_cfg
            best_score = metrics["score"]
            _save_best(best_config_path, best_cfg, best_score, i, metrics)

        result = IterationResult(
            iteration=i,
            mutation=mutation_desc,
            score=metrics["score"],
            pooled_log_loss=metrics["pooled_log_loss"],
            pooled_mrr=metrics["pooled_mrr"],
            n_features=metrics.get("n_features", 0),
            accepted=accepted,
            is_best=is_best,
            elapsed_s=round(elapsed, 2),
            config=candidate_cfg,
            window_results=metrics["windows"],
        )

        _append_result(results_path, result)

        status = "✓ KEEP" if accepted else "  skip"
        logger.info(
            "[%3d/%d] %s  score=%.4f (best=%.4f)  ll=%.4f  mrr=%.4f  feats=%d  %.1fs  %s",
            i + 1, start_iter + n_iterations,
            status,
            metrics["score"], best_score,
            metrics["pooled_log_loss"], metrics["pooled_mrr"],
            metrics.get("n_features", 0),
            elapsed,
            mutation_desc,
        )

    logger.info("\nDone. Best score: %.4f", best_score)
    logger.info("Results: %s", results_path)
    logger.info("Best config: %s", best_config_path)

    # Print final best config summary
    _print_diff(DEFAULT_CONFIG, best_cfg)


def _save_best(
    path: Path,
    cfg: dict[str, Any],
    score: float,
    iteration: int,
    metrics: dict[str, Any],
) -> None:
    with open(path, "w") as f:
        json.dump({
            "score": score,
            "iteration": iteration,
            "pooled_log_loss": metrics.get("pooled_log_loss"),
            "pooled_mrr": metrics.get("pooled_mrr"),
            "n_features": metrics.get("n_features"),
            "config": cfg,
        }, f, indent=2)


def _append_result(path: Path, result: IterationResult) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def _print_diff(base: dict[str, Any], best: dict[str, Any]) -> None:
    """Print parameters that differ from the baseline."""
    diffs: list[str] = []

    def _compare(b: Any, n: Any, prefix: str = "") -> None:
        if isinstance(b, dict):
            for k in b:
                _compare(b[k], n.get(k) if isinstance(n, dict) else None, f"{prefix}.{k}" if prefix else k)
        else:
            if b != n:
                diffs.append(f"  {prefix}: {b!r} → {n!r}")

    _compare(base, best)
    if diffs:
        logger.info("\nChanges from baseline:")
        for d in diffs:
            logger.info(d)
    else:
        logger.info("\nNo changes from baseline (baseline was already optimal)")


def load_config() -> dict[str, Any]:
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoresearch: iterative model improvement via random mutations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of mutations to try (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--windows",
        nargs="+",
        default=list(FESTIVAL_WINDOWS.keys()),
        choices=list(FESTIVAL_WINDOWS.keys()),
        help="Festival windows to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--out",
        default="data/autoresearch/run_001",
        help="Output directory for results.jsonl and best_config.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from best_config.json in --out if it exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_autoresearch(
        windows=args.windows,
        n_iterations=args.iterations,
        seed=args.seed,
        out_dir=PROJECT_ROOT / args.out,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
