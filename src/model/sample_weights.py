"""Helpers for optional sample weighting during model training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sample_weights_enabled(sample_weights_cfg: dict[str, Any] | None) -> bool:
    return bool(sample_weights_cfg and sample_weights_cfg.get("enabled", False))


def format_sample_weights_config(sample_weights_cfg: dict[str, Any] | None) -> str:
    if not sample_weights_enabled(sample_weights_cfg):
        return "disabled"

    strategy = str(sample_weights_cfg.get("strategy", "recency"))
    if strategy != "recency":
        return strategy

    recency_cfg = sample_weights_cfg.get("recency", {})
    half_life_days = float(recency_cfg.get("half_life_days", 365.0))
    min_weight = float(recency_cfg.get("min_weight", 0.35))
    normalize = bool(recency_cfg.get("normalize", True))
    return (
        f"recency(half_life_days={half_life_days:g}, "
        f"min_weight={min_weight:.2f}, normalize={normalize})"
    )


def summarise_sample_weights(weights: pd.Series) -> dict[str, float]:
    arr = weights.to_numpy(dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def build_sample_weights(
    train_df: pd.DataFrame,
    reference_date: pd.Timestamp | str,
    sample_weights_cfg: dict[str, Any] | None,
    date_col: str = "date",
) -> pd.Series | None:
    """Return per-row sample weights, or None when weighting is disabled."""
    if not sample_weights_enabled(sample_weights_cfg):
        return None

    strategy = str(sample_weights_cfg.get("strategy", "recency"))
    if strategy != "recency":
        raise ValueError(f"Unsupported sample-weight strategy: {strategy}")

    if date_col not in train_df.columns:
        raise KeyError(f"Date column {date_col!r} not found in training dataframe")

    recency_cfg = sample_weights_cfg.get("recency", {})
    half_life_days = float(recency_cfg.get("half_life_days", 365.0))
    min_weight = float(recency_cfg.get("min_weight", 0.35))
    normalize = bool(recency_cfg.get("normalize", True))

    if half_life_days <= 0:
        raise ValueError("recency.half_life_days must be > 0")
    if min_weight < 0 or min_weight > 1:
        raise ValueError("recency.min_weight must be within [0, 1]")

    dates = pd.to_datetime(train_df[date_col]).dt.normalize()
    ref = pd.Timestamp(reference_date).normalize()
    age_days = (ref - dates).dt.days.to_numpy(dtype=np.float64)
    age_days = np.clip(age_days, a_min=0.0, a_max=None)

    decay = np.exp2(-age_days / half_life_days)
    weights = min_weight + ((1.0 - min_weight) * decay)

    if normalize:
        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights = weights / mean_weight

    return pd.Series(weights, index=train_df.index, dtype=np.float64, name="sample_weight")
