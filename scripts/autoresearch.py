"""Autoresearch loop — agent-in-the-loop iterative model improvement.

An LLM agent analyses results, reasons about what to try, and proposes
config changes. Each proposal is evaluated on fixed festival windows.
Improvements are kept and fed back to the agent for the next iteration.

Primary metric : pooled log loss across all windows (lower = better)
Secondary metric: pooled MRR — mean reciprocal rank of the winner (higher = better)

The "score" used to decide keep/discard is:
    score = pooled_log_loss - 0.5 * pooled_mrr   (lower = better)

Usage:
    .venv/bin/python scripts/autoresearch.py --iterations 10
    .venv/bin/python scripts/autoresearch.py --iterations 50 --model anthropic/claude-sonnet-4-20250514
    .venv/bin/python scripts/autoresearch.py --iterations 20 --apply
    .venv/bin/python scripts/autoresearch.py --provider random --iterations 100

Each iteration appends one line to <out>/results.jsonl.
Best config is always written to <out>/best_config.json.
When --apply is set, best config is written back to configs/pipeline.yaml.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import log_loss

from src.model.feature_groups import (
    FEATURE_GROUPS,
    REQUIRED_FEATURE_GROUPS,
    active_feature_cols,
    default_feature_group_flags,
)
from src.model.sample_weights import build_sample_weights

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

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
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "path_smooth": 0.0,
        "verbose": -1,
        "seed": 42,
    },
    # Feature groups to include (all on by default)
    "feature_groups": default_feature_group_flags(),
    "sample_weights": {
        "enabled": False,
        "strategy": "recency",
        "recency": {
            "half_life_days": 365,
            "min_weight": 0.35,
            "normalize": True,
        },
    },
}

TUNEABLE_LGBM_KEYS = (
    "num_leaves",
    "learning_rate",
    "feature_fraction",
    "bagging_fraction",
    "min_child_samples",
    "lambda_l1",
    "lambda_l2",
    "path_smooth",
)

# ---------------------------------------------------------------------------
# Search space definition (shared with agent prompt)
# ---------------------------------------------------------------------------

SEARCH_SPACE: list[dict[str, Any]] = [
    {"path": "lgbm.num_leaves",        "type": "int",       "low": 16,   "high": 96},
    {"path": "lgbm.learning_rate",     "type": "log_float", "low": 0.01, "high": 0.15},
    {"path": "lgbm.feature_fraction",  "type": "float",     "low": 0.5,  "high": 1.0},
    {"path": "lgbm.bagging_fraction",  "type": "float",     "low": 0.5,  "high": 1.0},
    {"path": "lgbm.min_child_samples", "type": "int",       "low": 5,    "high": 50},
    {"path": "lgbm.lambda_l1",        "type": "log_float", "low": 0.001, "high": 5.0},
    {"path": "lgbm.lambda_l2",        "type": "log_float", "low": 0.001, "high": 5.0},
    {"path": "lgbm.path_smooth",      "type": "float",     "low": 0.0,  "high": 10.0},
    *[
        {"path": f"feature_groups.{g}", "type": "bool"}
        for g in FEATURE_GROUPS
        if g not in REQUIRED_FEATURE_GROUPS
    ],
]

# Legacy alias for random fallback
MUTATIONS = SEARCH_SPACE


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


def pipeline_baseline_config(pipeline_cfg: dict[str, Any]) -> dict[str, Any]:
    """Seed the search from the active pipeline config instead of the hardcoded default."""
    baseline_cfg = copy.deepcopy(DEFAULT_CONFIG)

    model_cfg = pipeline_cfg.get("model", {})
    lgbm_cfg = model_cfg.get("lgbm", {})
    for key in TUNEABLE_LGBM_KEYS:
        if key in lgbm_cfg:
            baseline_cfg["lgbm"][key] = lgbm_cfg[key]

    pipeline_groups = model_cfg.get("feature_groups", {})
    for group_name in baseline_cfg["feature_groups"]:
        if group_name in pipeline_groups:
            baseline_cfg["feature_groups"][group_name] = bool(pipeline_groups[group_name])

    if "sample_weights" in model_cfg:
        baseline_cfg["sample_weights"] = copy.deepcopy(model_cfg["sample_weights"])

    return baseline_cfg


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
    sample_weights_cfg: dict[str, Any] | None,
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

    train_weights = build_sample_weights(train_df, val_start, sample_weights_cfg)
    train_ds = lgb.Dataset(
        train_df[active_cols],
        label=train_df["won"],
        weight=train_weights.to_numpy(dtype=np.float64) if train_weights is not None else None,
    )
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
    feature_cols = active_feature_cols(all_feature_cols, cfg["feature_groups"])
    if len(feature_cols) == 0:
        return {"score": float("inf"), "pooled_log_loss": float("inf"), "pooled_mrr": 0.0, "windows": {}}

    window_results: dict[str, dict[str, float]] = {}
    for w in windows:
        window_results[w] = _eval_window(
            w,
            features,
            runners,
            feature_cols,
            cfg["lgbm"],
            cfg.get("sample_weights"),
        )

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
# Random mutation fallback (--provider random)
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
# Agent-in-the-loop: LLM proposes mutations
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert ML researcher optimising a LightGBM model that predicts \
horse racing winners at UK/Irish National Hunt festivals (Cheltenham, Aintree).

## Objective
Minimise the composite score:  score = pooled_log_loss - 0.5 * pooled_mrr  (lower = better)
- log_loss measures probability calibration (lower = better)
- MRR (mean reciprocal rank) measures how highly the winner is ranked (higher = better)

The model is evaluated via walk-forward backtests on 6 festival windows \
(3 Cheltenham + 3 Aintree festivals, 2023-2025). Each window trains on all \
data before the festival start date, then evaluates on that festival's races. \
Validation sets are small: ~300-450 runners across 20-28 races per window.

## Domain knowledge — horse racing prediction

### What the literature says works
- Market odds (Betfair SP) are the single strongest predictor (r>0.90 with actual \
win frequency). The model should complement the market, not replicate it.
- Official ratings (OR, RPR, Topspeed) are the strongest fundamental features.
- Going preference, distance suitability, and course form are Tier 1 features for NH racing.
- Trainer/jockey festival-specific strike rates carry genuine signal (Mullins, Henderson, Elliott).
- Pedigree features help most for lightly-raced novices; less useful for established form horses.
- Comment-derived NLP features have previously DEGRADED model performance in this system.
- The market features group can cause the model to "parrot" market prices, diluting its \
ability to find genuine value overlays. Watch whether toggling market off improves ROI-relevant metrics.

### What typically doesn't work
- Very complex trees on small datasets — overfitting is the primary risk
- Removing runner_profile or horse_form groups — these consistently hurt badly
- Individual feature-level removal is too noisy at this sample size; group-level toggles are appropriate

## Search space
You may change ONE parameter per iteration. Here are the tuneable parameters:

### LightGBM hyperparameters
- lgbm.num_leaves: int [16, 96] — tree complexity. Default: 31. Sweet spot: 16-40. \
Values >48 almost certainly overfit on our data size.
- lgbm.learning_rate: float [0.01, 0.15] (log-scale) — step size. Default: 0.05. \
Sweet spot: 0.02-0.05. Below 0.015, early stopping fires too early. \
Lower LR + more trees = better generalisation. LR and num_leaves interact: \
lower LR pairs well with slightly fewer leaves.
- lgbm.feature_fraction: float [0.5, 1.0] — column subsampling per tree. Default: 0.8. \
Recommended: 0.6-0.8. With correlated rating features (or/rpr/ts), lower values \
force tree diversity. Below 0.5 risks excluding critical features too often.
- lgbm.bagging_fraction: float [0.5, 1.0] — row subsampling per tree. Default: 0.8. \
Recommended: 0.7-0.85. Must be paired with bagging_freq > 0 (we use 5).
- lgbm.min_child_samples: int [5, 50] — min samples per leaf. Default: 20. \
With ~8% positive rate, leaves need 12-15+ samples for stable estimates. \
Range 15-35 is safest. Values <10 risk noisy leaves; >40 may be too coarse.
- lgbm.lambda_l1: float [0.001, 5.0] (log-scale) — L1 regularization on leaf weights. \
Default: 0. Adds sparsity. Start around 0.01-0.1, rarely helps above 1.0.
- lgbm.lambda_l2: float [0.001, 5.0] (log-scale) — L2 regularization on leaf weights. \
Default: 0. Smooths predictions. More commonly useful than L1. Try 0.01-1.0.
- lgbm.path_smooth: float [0, 10] — smoothing applied to tree predictions. \
Default: 0. Values 1-5 can help on small datasets by shrinking leaf outputs. \
Higher values = more conservative predictions.

### Feature group toggles (true=included, false=excluded)
- feature_groups.ratings: {ratings_cols} — official/performance ratings (Tier 1, usually keep)
- feature_groups.horse_form: {horse_form_cols} — form, win rates, history (Tier 1, usually keep)
- feature_groups.connections: {connections_cols} — trainer/jockey stats (Tier 2, usually additive)
- feature_groups.pedigree: {pedigree_cols} — sire performance (Tier 3, may be too sparse)
- feature_groups.market_proxy: {market_proxy_cols} — coarse market ordering only (useful but operationally simpler than full market features)
- feature_groups.runner_profile: {runner_profile_cols} — weight, class, course-distance profile (Tier 1-2, usually keep)
- feature_groups.market: {market_cols} — Betfair market data (powerful but may cause market-parroting)
- feature_groups.comments: {comments_cols} — NLP-derived comment features (previously degraded model)
- feature_groups.ratings_vs_field: {ratings_vs_field_cols} — within-race relative ratings (NEW — high signal expected)
- feature_groups.enhanced: {enhanced_cols} — sex, festival exp, class context (NEW — domain-driven)
- feature_groups.connections_extended: {connections_extended_cols} — trainer/jockey by class, course, race type (NEW — granular connection stats)
- feature_groups.horse_context: {horse_context_cols} — first-time flags, field quality, OR movement (NEW — situational context)

Note: "race_context" is always required and cannot be toggled off.

## Search strategy guidance

### Phase-based approach
- Iterations 0-10: Explore extremes of each continuous parameter. Test each feature toggle once.
- Iterations 10-30: Focus on parameters that showed sensitivity. Fine-tune promising regions.
- Iterations 30+: Small adjustments, interaction effects (e.g., LR + num_leaves together).

### Rules
- NEVER propose the exact same change that was previously rejected.
- Score improvements < 0.005 are likely noise with our validation sizes. Be sceptical of tiny gains.
- If a change improves some windows but hurts others, it may be overfitting.
- When scores are similar, prefer MORE regularisation (lower num_leaves, higher min_child_samples).
- If the last 5+ iterations were all rejected, try a larger/different type of change.
- Track patterns: if lowering LR keeps helping, keep going. If removing features keeps hurting, stop trying.
- early_stopping_rounds (50) * learning_rate should be ~1.0-2.0 for adequate convergence.

## Your task
Given the current best config and the history of all previous attempts, propose \
exactly ONE change. Think carefully about what the history tells you.

## Response format
Respond with valid JSON only, no markdown fencing. The JSON must have exactly these keys:
{{
  "reasoning": "Brief explanation of why you're proposing this change",
  "path": "the.param.path",
  "value": <the new value>
}}
"""


def _format_system_prompt() -> str:
    """Fill in feature column names for the system prompt."""
    return SYSTEM_PROMPT.format(
        ratings_cols=", ".join(FEATURE_GROUPS["ratings"]),
        horse_form_cols=", ".join(FEATURE_GROUPS["horse_form"][:5]) + ", ...",
        connections_cols=", ".join(FEATURE_GROUPS["connections"][:3]) + ", ...",
        pedigree_cols=", ".join(FEATURE_GROUPS["pedigree"]),
        market_proxy_cols=", ".join(FEATURE_GROUPS["market_proxy"]),
        runner_profile_cols=", ".join(FEATURE_GROUPS["runner_profile"]),
        market_cols=", ".join(FEATURE_GROUPS["market"]),
        comments_cols=", ".join(FEATURE_GROUPS["comments"]),
        ratings_vs_field_cols=", ".join(FEATURE_GROUPS["ratings_vs_field"]),
        enhanced_cols=", ".join(FEATURE_GROUPS["enhanced"]),
        connections_extended_cols=", ".join(FEATURE_GROUPS["connections_extended"]),
        horse_context_cols=", ".join(FEATURE_GROUPS["horse_context"]),
    )


def _format_history(history: list[dict[str, Any]], best_score: float) -> str:
    """Format trial history for the agent's user message."""
    if not history:
        return "No previous iterations yet. This is the first proposal."

    lines: list[str] = []
    for h in history:
        status = "ACCEPTED" if h["accepted"] else "rejected"
        lines.append(
            f"  iter {h['iteration']}: {h['mutation']} → "
            f"score={h['score']:.4f} ll={h['pooled_log_loss']:.4f} "
            f"mrr={h['pooled_mrr']:.4f} feats={h['n_features']} [{status}]"
        )

    accepted_count = sum(1 for h in history if h["accepted"])
    return (
        f"Trial history ({len(history)} iterations, {accepted_count} accepted):\n"
        + "\n".join(lines)
        + f"\n\nCurrent best score: {best_score:.4f}"
    )


def _build_user_message(
    current_cfg: dict[str, Any],
    best_score: float,
    history: list[dict[str, Any]],
) -> str:
    """Build the user message with current config + history."""
    # Show only the tuneable parts of the config
    tuneable_cfg = {
        "lgbm": {
            k: v for k, v in current_cfg["lgbm"].items()
            if k in {"num_leaves", "learning_rate", "feature_fraction", "bagging_fraction", "min_child_samples", "lambda_l1", "lambda_l2", "path_smooth"}
        },
        "feature_groups": current_cfg["feature_groups"],
        "sample_weights": current_cfg.get("sample_weights"),
    }

    return (
        f"Current best config (score={best_score:.4f}):\n"
        f"{json.dumps(tuneable_cfg, indent=2)}\n\n"
        f"{_format_history(history, best_score)}\n\n"
        "Propose ONE change. Respond with JSON only."
    )


def _create_openrouter_client(model: str) -> OpenAI:
    """Create an OpenAI-compatible client for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY env var is required.\n"
            "Get a key at https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _validate_proposal(proposal: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    """Validate that a proposal is within the search space."""
    path = proposal.get("path", "")
    value = proposal.get("value")

    if not path or value is None:
        return False, f"Missing 'path' or 'value' in proposal: {proposal}"

    # Check the path exists in search space
    matching = [s for s in SEARCH_SPACE if s["path"] == path]
    if not matching:
        return False, f"Unknown path: {path}. Must be one of: {[s['path'] for s in SEARCH_SPACE]}"

    spec = matching[0]

    # Validate value type and range
    if spec["type"] == "int":
        if not isinstance(value, (int, float)):
            return False, f"Expected int for {path}, got {type(value).__name__}"
        value = int(value)
        if value < spec["low"] or value > spec["high"]:
            return False, f"{path}={value} out of range [{spec['low']}, {spec['high']}]"
    elif spec["type"] in ("float", "log_float"):
        if not isinstance(value, (int, float)):
            return False, f"Expected float for {path}, got {type(value).__name__}"
        value = float(value)
        if value < spec["low"] or value > spec["high"]:
            return False, f"{path}={value} out of range [{spec['low']}, {spec['high']}]"
    elif spec["type"] == "bool":
        if not isinstance(value, bool):
            return False, f"Expected bool for {path}, got {type(value).__name__}"

    return True, ""


def agent_propose_mutation(
    cfg: dict[str, Any],
    best_score: float,
    history: list[dict[str, Any]],
    client: OpenAI,
    model: str,
) -> tuple[dict[str, Any], str]:
    """Ask the LLM agent to propose a config mutation. Returns (new_cfg, description)."""
    system_prompt = _format_system_prompt()
    user_message = _build_user_message(cfg, best_score, history)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    raw_text = response.choices[0].message.content or ""
    # Strip markdown fencing if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    proposal = json.loads(cleaned)

    path: str = proposal["path"]
    value: Any = proposal["value"]
    reasoning: str = proposal.get("reasoning", "")

    # Validate
    valid, err = _validate_proposal(proposal, cfg)
    if not valid:
        raise ValueError(f"Invalid proposal from agent: {err}")

    # Coerce types
    spec = [s for s in SEARCH_SPACE if s["path"] == path][0]
    if spec["type"] == "int":
        value = int(value)
    elif spec["type"] in ("float", "log_float"):
        value = float(value)

    # Apply mutation
    new_cfg = copy.deepcopy(cfg)
    old_value = _get(new_cfg, path)
    _set(new_cfg, path, value)

    desc = f"{path}: {old_value!r} → {value!r}"
    if reasoning:
        logger.info("  Agent reasoning: %s", reasoning)

    return new_cfg, desc


# ---------------------------------------------------------------------------
# Auto-apply: write best config back to pipeline.yaml
# ---------------------------------------------------------------------------

def apply_config_to_pipeline(cfg: dict[str, Any]) -> None:
    """Update configs/pipeline.yaml with the best lgbm params."""
    pipeline_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(pipeline_path) as f:
        pipeline = yaml.safe_load(f)

    # Update lgbm params (only the tuneable ones, preserve the rest)
    for key in ("num_leaves", "learning_rate", "feature_fraction", "bagging_fraction", "min_child_samples", "lambda_l1", "lambda_l2", "path_smooth"):
        if key in cfg["lgbm"]:
            pipeline["model"]["lgbm"][key] = cfg["lgbm"][key]

    pipeline["model"]["feature_groups"] = cfg["feature_groups"]
    if "sample_weights" in cfg:
        pipeline["model"]["sample_weights"] = cfg["sample_weights"]

    with open(pipeline_path, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

    logger.info("Updated configs/pipeline.yaml with best lgbm params")

    disabled = [g for g, enabled in cfg["feature_groups"].items() if not enabled]
    if disabled:
        logger.info("Updated pipeline feature groups; disabled: %s", disabled)


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
    provider: str,
    model: str,
    apply: bool,
) -> None:
    cfg = load_config()

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
    available_windows: list[str] = []
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

    # Set up provider
    client: OpenAI | None = None
    if provider == "openrouter":
        client = _create_openrouter_client(model)
        logger.info("Using OpenRouter agent: %s", model)
    else:
        logger.info("Using random mutations (no agent)")

    # Resume from previous best if requested
    history: list[dict[str, Any]] = []
    if resume and best_config_path.exists():
        with open(best_config_path) as f:
            saved = json.load(f)
        best_cfg = saved["config"]
        best_score = saved["score"]
        start_iter = saved.get("iteration", 0) + 1
        logger.info("Resuming from iteration %d, best score=%.4f", start_iter, best_score)
        # Load history from results.jsonl for agent context
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
            logger.info("Loaded %d historical trials for agent context", len(history))
    else:
        baseline_cfg = pipeline_baseline_config(cfg)
        best_cfg = copy.deepcopy(baseline_cfg)
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

        # Get proposal from agent or random fallback
        if provider == "openrouter" and client is not None:
            try:
                candidate_cfg, mutation_desc = agent_propose_mutation(
                    best_cfg, best_score, history, client, model,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("Agent proposal failed (%s), falling back to random", e)
                candidate_cfg, mutation_desc = sample_mutation(best_cfg, rng)
                mutation_desc = f"[random fallback] {mutation_desc}"
        else:
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

        # Add to history for agent context (compact version without full config)
        history.append({
            "iteration": i,
            "mutation": mutation_desc,
            "score": metrics["score"],
            "pooled_log_loss": metrics["pooled_log_loss"],
            "pooled_mrr": metrics["pooled_mrr"],
            "n_features": metrics.get("n_features", 0),
            "accepted": accepted,
        })

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
    baseline_for_diff = pipeline_baseline_config(cfg)
    _print_diff(baseline_for_diff, best_cfg)

    # Auto-apply if requested
    if apply:
        apply_config_to_pipeline(best_cfg)


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
        description="Autoresearch: agent-in-the-loop iterative model improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for random fallback")
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
    parser.add_argument(
        "--provider",
        choices=["openrouter", "random"],
        default="openrouter",
        help="Mutation provider: 'openrouter' (LLM agent) or 'random' (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4-6",
        help="Model ID for OpenRouter (default: anthropic/claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply best config to configs/pipeline.yaml when done",
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
        provider=args.provider,
        model=args.model,
        resume=args.resume,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
