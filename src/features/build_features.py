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
from src.features.groups.ratings_vs_field import ratings_vs_field
from src.features.groups.runner_profile import runner_profile
from src.features.groups.enhanced import enhanced
from src.features.groups.connections_extended import connections_extended
from src.features.groups.horse_context import horse_context
from src.features.racecard import load_racecards

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COURSE_TRACK_DIRECTION = {
    "aintree": "left",
    "ascot": "right",
    "catterick": "left",
    "cheltenham": "left",
    "chepstow": "left",
    "doncaster": "left",
    "exeter": "right",
    "fairyhouse": "right",
    "haydock": "left",
    "kempton": "right",
    "leopardstown": "left",
    "navan": "left",
    "newbury": "left",
    "punchestown": "right",
    "sandown": "right",
    "thurles": "right",
    "warwick": "left",
    "wetherby": "left",
    "wincanton": "right",
}

DOMINANT_STYLE_CODE_MAP = {
    "front-runner": 0,
    "prominent": 1,
    "mid-division": 2,
    "held-up": 3,
    "rear": 4,
}

COMMENT_FEATURE_COLUMNS = [
    "dominant_style_code",
    "pct_trouble",
    "pct_jumping_issues",
]


def load_config() -> dict:  # type: ignore[type-arg]
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _ensure_distance_m(races: pd.DataFrame) -> pd.DataFrame:
    """Ensure distance_m column exists, deriving from distance_f if needed."""
    if "distance_m" not in races.columns and "distance_f" in races.columns:
        races = races.copy()
        races["distance_m"] = (races["distance_f"] * 201.168).astype("Int64")
    return races


def _ensure_track_direction(races: pd.DataFrame) -> pd.DataFrame:
    """Ensure track_direction exists, using a curated per-course mapping."""
    races = races.copy()
    course_norm = races["course"].astype(str).str.strip().str.lower()
    mapped = course_norm.map(COURSE_TRACK_DIRECTION)

    if "track_direction" in races.columns:
        existing = races["track_direction"].astype("string").str.strip().str.lower()
        races["track_direction"] = existing.where(existing.notna() & (existing != ""), mapped)
    else:
        races["track_direction"] = mapped

    races.loc[races["track_direction"].isna(), "track_direction"] = pd.NA
    return races


def _normalise_key_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype("string").str.strip()
    out["horse_id"] = pd.to_numeric(out["horse_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["race_id", "horse_id"])
    out = out[out["race_id"] != ""]
    out["race_id"] = out["race_id"].astype(str)
    out["horse_id"] = out["horse_id"].astype(int)
    return out


def _load_comment_features(comment_path: Path) -> pd.DataFrame | None:
    """Load and normalise comment-derived runner features if available."""
    if not comment_path.exists():
        return None

    raw = pd.read_parquet(comment_path)
    if raw.empty:
        return None

    features = _normalise_key_types(raw)
    if "dominant_style_code" not in features.columns and "dominant_style" in features.columns:
        features["dominant_style_code"] = (
            features["dominant_style"].astype(str).str.strip().str.lower().map(DOMINANT_STYLE_CODE_MAP)
        )

    comment_cols = [c for c in COMMENT_FEATURE_COLUMNS if c in features.columns]
    if not comment_cols:
        return None

    keep = ["race_id", "horse_id"] + comment_cols
    features = features[keep].drop_duplicates(subset=["race_id", "horse_id"])
    return features


def _latest_comment_features_by_horse(
    comment_features: pd.DataFrame,
    historical_runners: pd.DataFrame,
) -> pd.DataFrame:
    """Build latest known comment-derived profile per horse for scoring races."""
    runners = _normalise_key_types(historical_runners[["race_id", "horse_id", "date"]])
    runners["race_date"] = pd.to_datetime(runners["date"], errors="coerce")
    merged = comment_features.merge(
        runners[["race_id", "horse_id", "race_date"]],
        on=["race_id", "horse_id"],
        how="left",
    )
    merged = merged.sort_values(["horse_id", "race_date", "race_id"])
    latest = merged.groupby("horse_id", as_index=False).tail(1)
    keep = ["horse_id"] + [c for c in COMMENT_FEATURE_COLUMNS if c in latest.columns]
    return latest[keep].drop_duplicates(subset=["horse_id"])


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
    races = _ensure_track_direction(races)

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

    logger.info("Computing ratings vs field features (G9)...")
    g9 = ratings_vs_field(con)
    logger.info("  → %d rows, %d features", len(g9), len(g9.columns) - 2)

    logger.info("Computing enhanced features (G10)...")
    g10 = enhanced(con)
    logger.info("  → %d rows, %d features", len(g10), len(g10.columns) - 2)

    logger.info("Computing connections extended features (G11)...")
    g11 = connections_extended(con)
    logger.info("  → %d rows, %d features", len(g11), len(g11.columns) - 2)

    logger.info("Computing horse context features (G12)...")
    g12 = horse_context(con)
    logger.info("  → %d rows, %d features", len(g12), len(g12.columns) - 2)

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

    # Comment-derived features (if comment parsing has been run)
    g6: pd.DataFrame | None = None
    comment_path = marts_dir / "comment_derived_features.parquet"
    comment_features = _load_comment_features(comment_path)
    if comment_features is not None:
        g6 = comment_features
        logger.info("Loaded comment-derived features (G6): %d rows, %d features", len(g6), len(g6.columns) - 2)

    # Base DataFrame: keys + metadata + labels
    base = runners[["race_id", "horse_id", "date", "course", "horse_name",
                     "finish_position"]].copy()
    fp = base["finish_position"]
    base["won"] = fp.fillna(0).eq(1).astype(int)  # type: ignore[union-attr]
    base["placed"] = fp.fillna(99).le(3).astype(int)  # type: ignore[union-attr]

    # Merge all feature groups
    features = base
    groups = [g5, g1, g2, g3, g4, g8, g9, g10, g11, g12]
    if g7 is not None:
        groups.append(g7)
    if g6 is not None:
        groups.append(g6)
    for group_df in groups:
        features = features.merge(group_df, on=["race_id", "horse_id"], how="left")

    logger.info("Final feature matrix: %d rows, %d columns", len(features), len(features.columns))

    # Write training features (all historical data)
    features.to_parquet(marts_dir / "features.parquet", index=False)
    logger.info("Written features.parquet")

    # Load racecards and write to parquet for scoring
    racecard_df = load_racecards()
    racecard_path = marts_dir / "racecard_runners.parquet"
    if racecard_df.empty:
        raise RuntimeError(
            "Racecard parsing returned zero rows after course filtering; "
            "refusing to reuse stale scoring inputs."
        )
    racecard_df.to_parquet(racecard_path, index=False)
    logger.info("Written racecard_runners.parquet (%d rows)", len(racecard_df))

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
    historical_races = _ensure_track_direction(historical_races)
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
    all_races = _ensure_track_direction(all_races)  # type: ignore[arg-type]

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
    g9 = ratings_vs_field(con)
    g10 = enhanced(con)
    g11 = connections_extended(con)
    g12 = horse_context(con)

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

    # Comment-derived features for scoring:
    # map each horse to latest known historical comment profile.
    g6_scoring: pd.DataFrame | None = None
    comment_path = marts_dir / "comment_derived_features.parquet"
    comment_features = _load_comment_features(comment_path)
    if comment_features is not None:
        latest_by_horse = _latest_comment_features_by_horse(comment_features, historical_runners)
        if not latest_by_horse.empty:
            g6_scoring = racecard[["race_id", "horse_id"]].copy()
            g6_scoring["race_id"] = g6_scoring["race_id"].astype(str)
            g6_scoring["horse_id"] = pd.to_numeric(g6_scoring["horse_id"], errors="coerce").astype("Int64")
            g6_scoring = g6_scoring.dropna(subset=["race_id", "horse_id"])
            g6_scoring["horse_id"] = g6_scoring["horse_id"].astype(int)
            g6_scoring = g6_scoring.drop_duplicates(subset=["race_id", "horse_id"])
            g6_scoring = g6_scoring.merge(latest_by_horse, on="horse_id", how="left")
            comment_cols = [c for c in COMMENT_FEATURE_COLUMNS if c in g6_scoring.columns]
            if comment_cols:
                logger.info(
                    "Comment feature coverage for scoring: %.1f%%",
                    100 * float(g6_scoring[comment_cols[0]].notna().mean()),
                )

    scoring_meta_cols = [
        "race_id", "horse_id", "date", "course", "horse_name",
        "race_name", "race_type", "race_class", "off_time",
        "distance_f", "pattern", "going", "official_rating",
    ]
    available_cols = [c for c in scoring_meta_cols if c in racecard.columns]
    base: pd.DataFrame = racecard[available_cols].copy()  # type: ignore[assignment]
    base["race_id"] = base["race_id"].astype(str)
    base["horse_id"] = pd.to_numeric(base["horse_id"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["horse_id"])
    base["horse_id"] = base["horse_id"].astype(int)

    features = base
    groups_2026 = [g5, g1, g2, g3, g4, g8, g9, g10, g11, g12]
    if g7_scoring is not None:
        groups_2026.append(g7_scoring)
    if g6_scoring is not None:
        groups_2026.append(g6_scoring)
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
        "Scoring feature coverage — or_current: %.1f%%, sp_rank: %.1f%%, market_implied_prob: %.1f%%, going_bucket: %.1f%%",
        100 * float(features_2026["or_current"].notna().mean()) if "or_current" in features_2026.columns else 0.0,
        100 * float(features_2026["sp_rank"].notna().mean()) if "sp_rank" in features_2026.columns else 0.0,
        100 * float(features_2026["market_implied_prob"].notna().mean()) if "market_implied_prob" in features_2026.columns else 0.0,
        100 * float(features_2026["going_bucket"].notna().mean()) if "going_bucket" in features_2026.columns else 0.0,
    )
    features_2026.to_parquet(marts_dir / "features_2026.parquet", index=False)
    logger.info("Written features_2026.parquet (%d rows)", len(features_2026))
