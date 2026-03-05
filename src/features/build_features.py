"""Feature engineering orchestrator — loads staged data, runs feature groups, writes marts.

Reads from data/staged/parquet/{races,runners}.parquet
Writes to data/marts/features.parquet (training) and data/marts/features_2026.parquet (scoring)
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from src.features.groups import ENRICHED_VIEW_SQL
from src.features.groups.connections import connections
from src.features.groups.horse_form import horse_form
from src.features.groups.pedigree import pedigree
from src.features.groups.race_context import race_context
from src.features.groups.market import market
from src.features.groups.ratings import ratings
from src.features.groups.runner_profile import runner_profile
from src.features.racecard import load_racecards

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:  # type: ignore[type-arg]
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _ensure_distance_m(races: pd.DataFrame) -> pd.DataFrame:
    """Ensure distance_m column exists, deriving from distance_f if needed."""
    if "distance_m" not in races.columns and "distance_f" in races.columns:
        races = races.copy()
        races["distance_m"] = (races["distance_f"] * 201.168).astype("Int64")
    return races


def build_features() -> None:
    """Build all feature groups and write to marts."""
    config = load_config()
    parquet_dir = PROJECT_ROOT / config["paths"]["staged_parquet"]
    marts_dir = PROJECT_ROOT / config["paths"]["marts"]
    marts_dir.mkdir(parents=True, exist_ok=True)

    # Load staged data
    races = pd.read_parquet(parquet_dir / "races.parquet")
    runners = pd.read_parquet(parquet_dir / "runners.parquet")
    races = _ensure_distance_m(races)

    logger.info("Loaded %d races, %d runners", len(races), len(runners))

    # Create DuckDB in-memory connection and register tables
    con = duckdb.connect()
    con.register("races", races)
    con.register("runners", runners)
    con.execute(ENRICHED_VIEW_SQL)

    # Build each feature group
    logger.info("Computing race context features (G5)...")
    g5 = race_context(con)
    logger.info("  → %d rows, %d features", len(g5), len(g5.columns) - 2)

    logger.info("Computing rating features (G1)...")
    g1 = ratings(con)
    logger.info("  → %d rows, %d features", len(g1), len(g1.columns) - 2)

    logger.info("Computing horse form features (G2)...")
    g2 = horse_form(con)
    logger.info("  → %d rows, %d features", len(g2), len(g2.columns) - 2)

    logger.info("Computing connection features (G3)...")
    g3 = connections(con)
    logger.info("  → %d rows, %d features", len(g3), len(g3.columns) - 2)

    logger.info("Computing pedigree features (G4)...")
    g4 = pedigree(con)
    logger.info("  → %d rows, %d features", len(g4), len(g4.columns) - 2)

    logger.info("Computing runner profile features (G8)...")
    g8 = runner_profile(con)
    logger.info("  → %d rows, %d features", len(g8), len(g8.columns) - 2)

    # Market features (G7) — requires betfair_historical.parquet
    betfair_path = parquet_dir / "betfair_historical.parquet"
    g7: pd.DataFrame | None = None
    if betfair_path.exists():
        betfair = pd.read_parquet(betfair_path)
        con.register("betfair_historical", betfair)
        logger.info("Computing market features (G7)...")
        g7 = market(con)
        logger.info("  → %d rows, %d features", len(g7), len(g7.columns) - 2)

    con.close()

    # Base DataFrame: keys + metadata + labels
    base = runners[["race_id", "horse_id", "date", "course", "horse_name",
                     "finish_position"]].copy()
    fp = base["finish_position"]
    base["won"] = fp.fillna(0).eq(1).astype(int)  # type: ignore[union-attr]
    base["placed"] = fp.fillna(99).le(3).astype(int)  # type: ignore[union-attr]

    # Merge all feature groups
    features = base
    groups = [g5, g1, g2, g3, g4, g8]
    if g7 is not None:
        groups.append(g7)
    for group_df in groups:
        features = features.merge(group_df, on=["race_id", "horse_id"], how="left")

    logger.info("Final feature matrix: %d rows, %d columns", len(features), len(features.columns))

    # Write training features (all historical data)
    features.to_parquet(marts_dir / "features.parquet", index=False)
    logger.info("Written features.parquet")

    # Load racecards and write to parquet for scoring
    racecard_df = load_racecards()
    racecard_path = marts_dir / "racecard_runners.parquet"
    if not racecard_df.empty:
        racecard_df.to_parquet(racecard_path, index=False)
        logger.info("Written racecard_runners.parquet (%d rows)", len(racecard_df))

    if racecard_path.exists():
        logger.info("Building 2026 scoring features from racecard data...")
        _build_scoring_features(racecard_path, races, runners, marts_dir)


def _build_scoring_features(
    racecard_path: Path,
    historical_races: pd.DataFrame,
    historical_runners: pd.DataFrame,
    marts_dir: Path,
) -> None:
    """Build features for 2026 racecard entries using historical data as context."""
    racecard = pd.read_parquet(racecard_path)
    if racecard.empty:
        logger.warning("Racecard file is empty; skipping scoring feature build")
        return

    # Keep key types consistent across all tables to avoid silent merge misses.
    racecard = racecard.copy()
    racecard["race_id"] = racecard["race_id"].astype(str)
    racecard["horse_id"] = pd.to_numeric(racecard["horse_id"], errors="coerce").astype("Int64")
    racecard = racecard.dropna(subset=["race_id", "horse_id"])
    racecard["horse_id"] = racecard["horse_id"].astype(int)

    historical_races = historical_races.copy()
    historical_runners = historical_runners.copy()
    historical_races["race_id"] = historical_races["race_id"].astype(str)
    historical_runners["race_id"] = historical_runners["race_id"].astype(str)
    historical_runners["horse_id"] = pd.to_numeric(
        historical_runners["horse_id"], errors="coerce",
    ).astype("Int64")
    historical_runners = historical_runners.dropna(subset=["horse_id"])
    historical_runners["horse_id"] = historical_runners["horse_id"].astype(int)

    # Market data can provide a pre-race odds proxy for sp_rank on upcoming cards.
    config = load_config()
    exchange_path = PROJECT_ROOT / config["paths"]["raw_betfair"] / "exchange_odds.parquet"
    if exchange_path.exists():
        exchange = pd.read_parquet(exchange_path)
        if {"race_id", "horse_id", "wap"}.issubset(exchange.columns):
            exchange = exchange[["race_id", "horse_id", "wap"]].copy()
            exchange["race_id"] = exchange["race_id"].astype(str)
            exchange["horse_id"] = pd.to_numeric(exchange["horse_id"], errors="coerce").astype("Int64")
            exchange = exchange.dropna(subset=["horse_id"])
            exchange["horse_id"] = exchange["horse_id"].astype(int)
            exchange = exchange.rename(columns={"wap": "sp_decimal"})  # type: ignore[call-overload]
            racecard = racecard.merge(exchange, on=["race_id", "horse_id"], how="left")
            logger.info(
                "Injected exchange odds into racecard rows: %.1f%% coverage",
                100 * float(racecard["sp_decimal"].notna().mean()),
            )

    # Build a race-level table from racecard rows (one row per race_id).
    race_level_cols = [
        "race_id", "date", "course", "off_time", "race_name",
        "race_type", "race_class", "pattern", "going",
        "distance_f", "distance_m", "field_size", "is_handicap",
    ]
    racecard_races = racecard[[c for c in race_level_cols if c in racecard.columns]].copy()
    racecard_races = racecard_races.drop_duplicates(subset=["race_id"])
    if "race_class" in racecard_races.columns:
        racecard_races = racecard_races.rename(columns={"race_class": "class"})  # type: ignore[call-overload]

    # Align racecard races/runners to historical schemas for feature group SQL.
    for col in historical_races.columns:
        if col not in racecard_races.columns:
            racecard_races[col] = pd.NA
    racecard_races = racecard_races[historical_races.columns]

    racecard_runners = racecard.copy()
    for col in historical_runners.columns:
        if col not in racecard_runners.columns:
            racecard_runners[col] = pd.NA
    racecard_runners = racecard_runners[historical_runners.columns]

    all_races = pd.concat([historical_races, racecard_races], ignore_index=True)  # type: ignore[assignment]
    all_runners = pd.concat([historical_runners, racecard_runners], ignore_index=True)  # type: ignore[assignment]

    all_races = _ensure_distance_m(all_races)  # type: ignore[arg-type]

    con = duckdb.connect()
    con.register("races", all_races)
    con.register("runners", all_runners)
    con.execute(ENRICHED_VIEW_SQL)

    g5 = race_context(con)
    g1 = ratings(con)
    g2 = horse_form(con)
    g3 = connections(con)
    g4 = pedigree(con)
    g8 = runner_profile(con)

    # Market features for scoring — load betfair + exchange odds
    parquet_dir = PROJECT_ROOT / config["paths"]["staged_parquet"]
    betfair_path = parquet_dir / "betfair_historical.parquet"

    g7_scoring: pd.DataFrame | None = None
    for bf_path in [exchange_path, betfair_path]:
        if bf_path.exists():
            bf = pd.read_parquet(bf_path)
            if {"race_id", "horse_id"}.issubset(bf.columns):
                bf = bf.copy()
                bf["race_id"] = bf["race_id"].astype(str)
                bf["horse_id"] = pd.to_numeric(bf["horse_id"], errors="coerce").astype("Int64")
                bf = bf.dropna(subset=["horse_id"])
                bf["horse_id"] = bf["horse_id"].astype(int)
            con.register("betfair_historical", bf)
            g7_scoring = market(con)
            break

    con.close()

    scoring_meta_cols = [
        "race_id", "horse_id", "date", "course", "horse_name",
        "race_name", "race_type", "race_class", "off_time",
        "distance_f", "pattern", "official_rating",
    ]
    available_cols = [c for c in scoring_meta_cols if c in racecard.columns]
    base: pd.DataFrame = racecard[available_cols].copy()  # type: ignore[assignment]
    base["race_id"] = base["race_id"].astype(str)
    base["horse_id"] = pd.to_numeric(base["horse_id"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["horse_id"])
    base["horse_id"] = base["horse_id"].astype(int)

    features = base
    groups_2026 = [g5, g1, g2, g3, g4, g8]
    if g7_scoring is not None:
        groups_2026.append(g7_scoring)
    for group_df in groups_2026:
        group_df = group_df.copy()
        group_df["race_id"] = group_df["race_id"].astype(str)
        group_df["horse_id"] = pd.to_numeric(group_df["horse_id"], errors="coerce").astype("Int64")
        group_df = group_df.dropna(subset=["horse_id"])
        group_df["horse_id"] = group_df["horse_id"].astype(int)
        group_df = group_df.drop_duplicates(subset=["race_id", "horse_id"])
        features = features.merge(group_df, on=["race_id", "horse_id"], how="left")

    features_2026: pd.DataFrame = features.drop_duplicates(subset=["race_id", "horse_id"]).copy()  # type: ignore[assignment]

    logger.info(
        "Scoring feature coverage — or_current: %.1f%%, sp_rank: %.1f%%, market_implied_prob: %.1f%%",
        100 * float(features_2026["or_current"].notna().mean()) if "or_current" in features_2026.columns else 0.0,
        100 * float(features_2026["sp_rank"].notna().mean()) if "sp_rank" in features_2026.columns else 0.0,
        100 * float(features_2026["market_implied_prob"].notna().mean()) if "market_implied_prob" in features_2026.columns else 0.0,
    )
    features_2026.to_parquet(marts_dir / "features_2026.parquet", index=False)
    logger.info("Written features_2026.parquet (%d rows)", len(features_2026))
