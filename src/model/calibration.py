"""Helpers for calibrating per-runner win probabilities before race normalisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

EPSILON = 1e-6


def clip_probs(probs: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Keep probabilities away from 0/1 so logit-based transforms stay finite."""
    arr = np.asarray(probs, dtype=np.float64)
    return np.clip(arr, EPSILON, 1.0 - EPSILON)


def apply_temperature(
    probs: np.ndarray[Any, np.dtype[np.floating[Any]]],
    temperature: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Temperature-scale probabilities by operating on their logits."""
    safe = clip_probs(probs)
    t = max(float(temperature), EPSILON)
    logits = np.log(safe / (1.0 - safe))
    scaled_logits = logits / t
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def normalise_per_race(
    race_ids: pd.Series,
    probs: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> pd.Series:
    """Renormalise runner probabilities so each race sums to 1."""
    df = pd.DataFrame({"race_id": race_ids.to_numpy(), "prob": np.asarray(probs, dtype=np.float64)})
    totals = df.groupby("race_id")["prob"].transform("sum")
    totals = totals.clip(lower=EPSILON)
    return (df["prob"] / totals).astype(np.float64)


def fit_temperature_scaler(
    labels: pd.Series,
    race_ids: pd.Series,
    probs: np.ndarray[Any, np.dtype[np.floating[Any]]],
    min_temperature: float = 0.6,
    max_temperature: float = 2.0,
    num_grid_points: int = 71,
) -> dict[str, float | bool | str]:
    """Fit a 1-parameter temperature scaler against race-normalised log loss."""
    base_probs = clip_probs(probs)
    baseline_win_prob = normalise_per_race(race_ids, base_probs)
    baseline_loss = float(log_loss(labels, clip_probs(baseline_win_prob.to_numpy())))

    best_temperature = 1.0
    best_loss = baseline_loss

    grid = np.linspace(min_temperature, max_temperature, max(num_grid_points, 2))
    for temperature in grid:
        scaled_probs = apply_temperature(base_probs, float(temperature))
        win_prob = normalise_per_race(race_ids, scaled_probs)
        loss = float(log_loss(labels, clip_probs(win_prob.to_numpy())))
        if loss < best_loss - 1e-9:
            best_loss = loss
            best_temperature = float(temperature)

    return {
        "enabled": True,
        "method": "temperature",
        "selected": abs(best_temperature - 1.0) > 1e-9,
        "temperature": float(best_temperature),
        "log_loss_before": baseline_loss,
        "log_loss_after": best_loss,
    }


def load_calibration_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data.get("enabled", False):
        return None
    return data


def apply_calibration(
    probs: np.ndarray[Any, np.dtype[np.floating[Any]]],
    artifact: dict[str, Any] | None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Apply the saved calibration transform, or identity if unavailable."""
    if not artifact:
        return clip_probs(probs)
    method = artifact.get("method")
    if method != "temperature":
        return clip_probs(probs)
    temperature = float(artifact.get("temperature", 1.0))
    return apply_temperature(probs, temperature)
