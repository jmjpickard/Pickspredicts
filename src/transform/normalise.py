"""Normalise raw rpscrape CSVs and racecard JSONs into DuckDB + Parquet.

Reads raw CSV files from data/raw/results/ and JSON from data/raw/racecards/,
maps to canonical schema, deduplicates, and writes to DuckDB + Parquet.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Non-completion position codes from rpscrape
NON_COMPLETION_CODES = {"F", "PU", "UR", "BD", "CO", "RR", "RO", "SU", "DSQ", "VOI", "REF"}


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_position(pos: str) -> int | None:
    """Parse finishing position. Returns None for non-completions (F, PU, UR, etc.)."""
    if pd.isna(pos):
        return None
    pos_str = str(pos).strip()
    try:
        return int(pos_str)
    except ValueError:
        return None


def safe_float(val: object) -> float | None:
    """Convert to float, returning None for missing/invalid values."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if s in ("", "-"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def safe_int(val: object) -> int | None:
    """Convert to int, returning None for missing/invalid values."""
    f = safe_float(val)
    if f is None:
        return None
    return int(f)


def is_handicap(race_class: str | None, rating_band: str | None, race_name: str | None) -> bool:
    """Determine if a race is a handicap from class, rating band, and name."""
    if rating_band and str(rating_band).strip() not in ("", "-"):
        return True
    if race_name and "handicap" in str(race_name).lower():
        return True
    return False


def normalise_race_type(raw_type: str | None) -> str | None:
    """Map rpscrape race type to canonical enum."""
    if raw_type is None:
        return None
    mapping = {
        "chase": "chase",
        "hurdle": "hurdle",
        "nh flat": "nh_flat",
        "flat": "flat",
    }
    return mapping.get(str(raw_type).strip().lower())


def load_results_csvs(raw_dir: Path) -> pd.DataFrame:
    """Load all results CSVs from raw_dir into a single DataFrame."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", raw_dir)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        logger.info("Loading %s", csv_file.name)
        df = pd.read_csv(csv_file, dtype=str)
        df["_source_file"] = csv_file.name
        frames.append(df)
        logger.info("  %d rows", len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total raw rows: %d", len(combined))
    return combined


def build_races_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract and deduplicate races table from raw data."""
    if raw.empty:
        return pd.DataFrame()

    # Each row in the CSV is one runner — extract unique race records
    race_cols_map = {
        "race_id": "race_id",
        "date": "date",
        "region": "region",
        "course_id": "course_id",
        "course": "course",
        "off": "off_time",
        "race_name": "race_name",
        "type": "race_type",
        "class": "class",
        "pattern": "pattern",
        "dist_f": "distance_f",
        "dist_m": "distance_m",
        "dist_y": "distance_y",
        "going": "going",
        "surface": "surface",
        "ran": "field_size",
        "age_band": "age_band",
        "rating_band": "rating_band",
        "sex_rest": "sex_rest",
    }

    available_cols = {k: v for k, v in race_cols_map.items() if k in raw.columns}
    races = raw[list(available_cols.keys())].copy()
    races = races.rename(columns=available_cols)  # type: ignore[call-overload]

    # Deduplicate by race_id (or date+course+off_time if no race_id)
    if "race_id" in races.columns:
        races = races.drop_duplicates(subset=["race_id"])
    else:
        dedup_cols = [c for c in ["date", "course", "off_time"] if c in races.columns]
        if dedup_cols:
            races = races.drop_duplicates(subset=dedup_cols)

    # Type conversions
    if "distance_f" in races.columns:
        races["distance_f"] = races["distance_f"].apply(safe_float)
    if "distance_m" in races.columns:
        races["distance_m"] = races["distance_m"].apply(safe_int)
    if "distance_y" in races.columns:
        races["distance_y"] = races["distance_y"].apply(safe_int)
    if "field_size" in races.columns:
        races["field_size"] = races["field_size"].apply(safe_int)
    if "course_id" in races.columns:
        races["course_id"] = races["course_id"].apply(safe_int)

    # Normalise race type
    if "race_type" in races.columns:
        races["race_type"] = races["race_type"].apply(normalise_race_type)

    # Derive is_handicap
    races["is_handicap"] = races.apply(
        lambda r: is_handicap(
            r.get("class"),
            r.get("rating_band"),
            r.get("race_name"),
        ),
        axis=1,
    )

    return races.reset_index(drop=True)


def build_runners_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract runners table from raw data."""
    if raw.empty:
        return pd.DataFrame()

    runner_cols_map = {
        "race_id": "race_id",
        "date": "date",
        "course": "course",
        "off": "off_time",
        "horse_id": "horse_id",
        "horse": "horse_name",
        "age": "age",
        "sex": "sex",
        "pos": "finish_position_raw",
        "ovr_btn": "ovr_btn",
        "btn": "btn",
        "lbs": "weight_lbs",
        "hg": "headgear",
        "secs": "time_secs",
        "sp": "sp_fractional",
        "dec": "sp_decimal",
        "or": "official_rating",
        "rpr": "rpr",
        "ts": "topspeed",
        "jockey_id": "jockey_id",
        "jockey": "jockey",
        "trainer_id": "trainer_id",
        "trainer": "trainer",
        "owner_id": "owner_id",
        "sire_id": "sire_id",
        "sire": "sire",
        "dam_id": "dam_id",
        "dam": "dam",
        "damsire_id": "damsire_id",
        "damsire": "damsire",
        "comment": "comment",
        "prize": "prize",
    }

    available_cols = {k: v for k, v in runner_cols_map.items() if k in raw.columns}
    runners = raw[list(available_cols.keys())].copy()
    runners = runners.rename(columns=available_cols)  # type: ignore[call-overload]

    # Parse finish position — keep raw for non-completion detection
    if "finish_position_raw" in runners.columns:
        runners["finish_position"] = runners["finish_position_raw"].apply(parse_position)
        runners = runners.drop(columns=["finish_position_raw"])

    # Numeric conversions
    for col in ["horse_id", "age", "weight_lbs", "official_rating", "rpr", "topspeed",
                 "jockey_id", "trainer_id", "owner_id", "sire_id", "dam_id", "damsire_id"]:
        if col in runners.columns:
            runners[col] = runners[col].apply(safe_int)

    for col in ["sp_decimal", "time_secs", "ovr_btn", "btn", "prize"]:
        if col in runners.columns:
            runners[col] = runners[col].apply(safe_float)

    # Deduplicate (same runner in same race from overlapping fetches)
    dedup_cols = []
    if "race_id" in runners.columns:
        dedup_cols.append("race_id")
    elif "date" in runners.columns and "course" in runners.columns and "off_time" in runners.columns:
        dedup_cols.extend(["date", "course", "off_time"])
    if "horse_id" in runners.columns:
        dedup_cols.append("horse_id")
    elif "horse_name" in runners.columns:
        dedup_cols.append("horse_name")

    if dedup_cols:
        runners = runners.drop_duplicates(subset=dedup_cols)

    return runners.reset_index(drop=True)


def build_betfair_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract Betfair historical pricing data from raw data."""
    if raw.empty:
        return pd.DataFrame()

    betfair_cols_map = {
        "race_id": "race_id",
        "date": "date",
        "course": "course",
        "off": "off_time",
        "horse_id": "horse_id",
        "horse": "horse_name",
        "bsp": "bsp",
        "wap": "wap",
        "morning_wap": "morning_wap",
        "pre_min": "pre_min",
        "pre_max": "pre_max",
        "ip_min": "ip_min",
        "ip_max": "ip_max",
        "morning_vol": "morning_vol",
        "pre_vol": "pre_vol",
        "ip_vol": "ip_vol",
    }

    available_cols = {k: v for k, v in betfair_cols_map.items() if k in raw.columns}

    # Check if any actual betfair data columns are present
    betfair_data_cols = {"bsp", "wap", "morning_wap", "pre_min", "pre_max",
                         "ip_min", "ip_max", "morning_vol", "pre_vol", "ip_vol"}
    present_data_cols = betfair_data_cols & set(raw.columns)
    if not present_data_cols:
        logger.info("No Betfair columns found in raw data")
        return pd.DataFrame()

    betfair = raw[list(available_cols.keys())].copy()
    betfair = betfair.rename(columns=available_cols)  # type: ignore[call-overload]

    # Convert numeric columns
    for col in ["bsp", "wap", "morning_wap", "pre_min", "pre_max",
                 "ip_min", "ip_max", "morning_vol", "pre_vol", "ip_vol"]:
        if col in betfair.columns:
            betfair[col] = betfair[col].apply(safe_float)

    for col in ["horse_id"]:
        if col in betfair.columns:
            betfair[col] = betfair[col].apply(safe_int)

    # Drop rows where all betfair price columns are null
    price_cols = [c for c in ["bsp", "wap", "pre_min", "pre_max"] if c in betfair.columns]
    if price_cols:
        betfair = betfair.dropna(subset=price_cols, how="all")

    # Deduplicate
    dedup_cols = []
    if "race_id" in betfair.columns:
        dedup_cols.append("race_id")
    if "horse_id" in betfair.columns:
        dedup_cols.append("horse_id")
    if dedup_cols:
        betfair = betfair.drop_duplicates(subset=dedup_cols)

    return betfair.reset_index(drop=True)


def write_to_duckdb(
    db_path: Path,
    races: pd.DataFrame,
    runners: pd.DataFrame,
    betfair: pd.DataFrame,
) -> None:
    """Write all tables to a DuckDB database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        if not races.empty:
            con.execute("DROP TABLE IF EXISTS races")
            con.execute("CREATE TABLE races AS SELECT * FROM races")
            logger.info("Written %d races to DuckDB", len(races))

        if not runners.empty:
            con.execute("DROP TABLE IF EXISTS runners")
            con.execute("CREATE TABLE runners AS SELECT * FROM runners")
            logger.info("Written %d runners to DuckDB", len(runners))

        if not betfair.empty:
            con.execute("DROP TABLE IF EXISTS betfair_historical")
            con.execute("CREATE TABLE betfair_historical AS SELECT * FROM betfair")
            logger.info("Written %d betfair rows to DuckDB", len(betfair))
    finally:
        con.close()


def write_to_parquet(
    parquet_dir: Path,
    races: pd.DataFrame,
    runners: pd.DataFrame,
    betfair: pd.DataFrame,
) -> None:
    """Write all tables to Parquet files."""
    parquet_dir.mkdir(parents=True, exist_ok=True)

    if not races.empty:
        races.to_parquet(parquet_dir / "races.parquet", index=False)
        logger.info("Written races.parquet (%d rows)", len(races))

    if not runners.empty:
        runners.to_parquet(parquet_dir / "runners.parquet", index=False)
        logger.info("Written runners.parquet (%d rows)", len(runners))

    if not betfair.empty:
        betfair.to_parquet(parquet_dir / "betfair_historical.parquet", index=False)
        logger.info("Written betfair_historical.parquet (%d rows)", len(betfair))


def run_validation(db_path: Path) -> None:
    """Run validation queries against the DuckDB database."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        queries = [
            ("Total races", "SELECT COUNT(*) FROM races"),
            ("Cheltenham races", "SELECT COUNT(*) FROM races WHERE course = 'Cheltenham'"),
            ("Total runners", "SELECT COUNT(*) FROM runners"),
            ("Distinct race types", "SELECT DISTINCT race_type FROM races ORDER BY race_type"),
        ]

        # Only run betfair queries if the table exists
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        if "betfair_historical" in tables:
            queries.append(
                ("BSP coverage", "SELECT COUNT(*) FROM betfair_historical WHERE bsp IS NOT NULL")
            )

        for label, query in queries:
            result = con.execute(query).fetchall()
            logger.info("Validation — %s: %s", label, result)
    finally:
        con.close()


def normalise() -> None:
    """Main normalisation pipeline: raw CSVs → DuckDB + Parquet."""
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_results"]
    db_path = PROJECT_ROOT / config["paths"]["staged_db"]
    parquet_dir = PROJECT_ROOT / config["paths"]["staged_parquet"]

    # Load all raw CSVs
    raw = load_results_csvs(raw_dir)
    if raw.empty:
        logger.error("No data to normalise")
        return

    logger.info("Raw columns: %s", list(raw.columns))

    # Build canonical tables
    races = build_races_df(raw)
    runners = build_runners_df(raw)
    betfair = build_betfair_df(raw)

    logger.info("Races: %d, Runners: %d, Betfair: %d", len(races), len(runners), len(betfair))

    # Write outputs
    write_to_duckdb(db_path, races, runners, betfair)
    write_to_parquet(parquet_dir, races, runners, betfair)

    # Validate
    run_validation(db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    normalise()
