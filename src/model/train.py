"""Train a LightGBM win-probability model and evaluate on Cheltenham 2025.

Usage:
    python -m src.pipeline --step train
"""

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)

matplotlib.use("Agg")

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


def _favourite_baseline(
    val: pd.DataFrame, runners: pd.DataFrame
) -> dict[str, float]:
    """Compute favourite (lowest SP) baseline metrics on validation set."""
    sp = runners[["race_id", "horse_id", "sp_decimal"]].copy()
    merged = val.merge(sp, on=["race_id", "horse_id"], how="left")

    results: dict[str, float] = {}

    # Top-1 accuracy: % of races where favourite won
    fav_idx = merged.groupby("race_id")["sp_decimal"].idxmin()
    fav_picks = merged.loc[fav_idx]
    fav_wins = int((fav_picks["won"] == 1).sum())
    total_races = int(fav_picks["race_id"].nunique())
    results["fav_top1_accuracy"] = fav_wins / total_races if total_races > 0 else 0.0

    # Flat-stake ROI at SP
    fav_picks = fav_picks.copy()
    fav_pnl: pd.Series[float] = fav_picks.apply(  # type: ignore[assignment]
        lambda r: (r["sp_decimal"] - 1) if r["won"] == 1 else -1, axis=1
    )
    results["fav_roi"] = float(fav_pnl.sum() / len(fav_pnl)) if len(fav_pnl) > 0 else 0.0
    results["fav_pnl"] = float(fav_pnl.sum())

    return results


def train() -> None:
    """Train LightGBM model, evaluate on Cheltenham 2025, save artifacts."""
    config = load_config()
    model_cfg = config["model"]
    lgbm_params: dict[str, Any] = model_cfg["lgbm"]
    output_dir = PROJECT_ROOT / model_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    features_path = PROJECT_ROOT / config["paths"]["marts"] / "features.parquet"
    runners_path = PROJECT_ROOT / config["paths"]["staged_parquet"] / "runners.parquet"

    df = pd.read_parquet(features_path)
    runners = pd.read_parquet(runners_path)
    logger.info("Loaded features: %s", df.shape)

    # --- Feature columns ---
    feature_cols = [c for c in df.columns if c not in META_COLS]
    label_col: str = model_cfg["label"]
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

    # --- Walk-forward split ---
    df["date"] = pd.to_datetime(df["date"])
    val_start = pd.Timestamp(model_cfg["val_start"])
    val_end = pd.Timestamp(model_cfg["val_end"])
    val_course: str = model_cfg["val_course"]

    train_mask = df["date"] < val_start
    val_mask = (
        (df["course"] == val_course)
        & (df["date"] >= val_start)
        & (df["date"] <= val_end)
    )

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(
        "Train: %d rows (%.1f%% positive), Val: %d rows (%.1f%% positive)",
        len(train_df),
        100 * float(train_df[label_col].mean()),  # type: ignore[arg-type]
        len(val_df),
        100 * float(val_df[label_col].mean()),  # type: ignore[arg-type]
    )
    n_val_races = val_df["race_id"].nunique()  # type: ignore[union-attr]
    logger.info("Val races: %d", n_val_races)

    # --- Training ---
    train_params = {k: v for k, v in lgbm_params.items() if k != "early_stopping_rounds"}

    train_ds = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
    val_ds = lgb.Dataset(val_df[feature_cols], label=val_df[label_col], reference=train_ds)

    callbacks = [
        lgb.early_stopping(lgbm_params["early_stopping_rounds"]),
        lgb.log_evaluation(50),
    ]

    booster = lgb.train(
        train_params,
        train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info("Best iteration: %d", booster.best_iteration)

    # --- Evaluation ---
    val_df["raw_prob"] = booster.predict(val_df[feature_cols], num_iteration=booster.best_iteration)
    val_df["win_prob"] = _softmax_per_race(val_df, "raw_prob")  # type: ignore[arg-type]

    # Log loss (on softmax-normalised probs)
    ll = float(log_loss(val_df[label_col], val_df["win_prob"]))
    logger.info("Validation log loss (softmax-normalised): %.4f", ll)

    # Top-1 accuracy: model's top pick per race
    top_idx = val_df.groupby("race_id")["win_prob"].idxmax()
    top_picks = val_df.loc[top_idx]
    model_wins = int((top_picks["won"] == 1).sum())
    total_races = int(top_picks["race_id"].nunique())
    model_top1 = model_wins / total_races if total_races > 0 else 0.0
    logger.info("Model top-1 accuracy: %.1f%% (%d/%d)", 100 * model_top1, model_wins, total_races)

    # ROI: flat-stake on model's top pick
    sp = runners[["race_id", "horse_id", "sp_decimal"]].copy()
    top_with_sp = top_picks.merge(sp, on=["race_id", "horse_id"], how="left")
    model_pnl: pd.Series[float] = top_with_sp.apply(  # type: ignore[assignment]
        lambda r: (r["sp_decimal"] - 1) if r["won"] == 1 else -1, axis=1
    )
    model_roi = float(model_pnl.sum() / len(model_pnl)) if len(model_pnl) > 0 else 0.0
    logger.info("Model ROI (top pick at SP): %.1f%% (PnL: %.1f)", 100 * model_roi, float(model_pnl.sum()))

    # Favourite baseline
    fav_metrics = _favourite_baseline(val_df, runners)  # type: ignore[arg-type]
    logger.info(
        "Favourite top-1: %.1f%%, ROI: %.1f%%",
        100 * fav_metrics["fav_top1_accuracy"],
        100 * fav_metrics["fav_roi"],
    )

    # Calibration plot
    prob_true, prob_pred = calibration_curve(
        val_df[label_col], val_df["win_prob"], n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration plot — Cheltenham 2025 validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)
    logger.info("Saved calibration plot")

    # Feature importance
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info("Top 10 features:\n%s", importance.head(10).to_string(index=False))

    # --- Save artifacts ---
    booster.save_model(str(output_dir / "model.txt"))
    logger.info("Saved model to %s", output_dir / "model.txt")

    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    metrics: dict[str, float | int] = {
        "log_loss": ll,
        "model_top1_accuracy": model_top1,
        "model_roi": model_roi,
        "model_pnl": float(model_pnl.sum()),
        "total_val_races": total_races,
        "total_val_runners": len(val_df),
        "best_iteration": booster.best_iteration,
        **fav_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete. Metrics: %s", json.dumps(metrics, indent=2))
