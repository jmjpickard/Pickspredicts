"""Fetch historical Betfair BSP data from free promo CSV endpoint.

Downloads dwbfprices CSVs for each race date in our dataset,
matches runners by (date, off_time, horse_name) with fuzzy matching,
and outputs betfair_historical.parquet in the same schema as normalise.build_betfair_df().

Usage:
    python -m src.pipeline --step fetch-betfair
"""

import io
import logging
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BSP_URL = "https://promo.betfair.com/betfairsp/prices/dwbfprices{region}win{date}.csv"

COURSE_TO_REGION: dict[str, str] = {
    "Cheltenham": "uk",
    "Aintree": "uk",
    "Punchestown": "ire",
    "Leopardstown": "ire",
}

# Minimum similarity ratio for fuzzy name matching
MATCH_THRESHOLD = 0.70


def load_config() -> dict:  # type: ignore[type-arg]
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _clean_name(name: str) -> str:
    """Strip country suffix and normalise for matching.

    'Constitution Hill (IRE)' → 'constitution hill'
    """
    cleaned = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name)
    return cleaned.strip().lower()


def _best_match(
    bsp_name: str, candidates: list[str],
) -> tuple[str | None, float]:
    """Find best fuzzy match for a BSP name among candidate runner names."""
    clean_bsp = _clean_name(bsp_name)
    best_name: str | None = None
    best_ratio = 0.0

    for candidate in candidates:
        clean_candidate = _clean_name(candidate)
        ratio = SequenceMatcher(None, clean_bsp, clean_candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = candidate

    if best_ratio >= MATCH_THRESHOLD:
        return best_name, best_ratio
    return None, best_ratio


def _fetch_csv(region: str, date: datetime, session: requests.Session) -> pd.DataFrame | None:
    """Fetch a single BSP CSV. Returns None on 404 or error."""
    date_str = date.strftime("%d%m%Y")
    url = BSP_URL.format(region=region, date=date_str)

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 520):
            logger.warning("Rate limited (%d) for %s, retrying in 5s...", resp.status_code, url)
            time.sleep(5)
            resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    if not resp.text.strip():
        return None

    try:
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        logger.warning("Failed to parse CSV from %s: %s", url, e)
        return None


def _parse_event_dt(event_dt: str) -> tuple[str, str]:
    """Parse 'DD-MM-YYYY HH:MM' → (date_str 'YYYY-MM-DD', off_time 'HH:MM')."""
    dt = datetime.strptime(event_dt.strip(), "%d-%m-%Y %H:%M")
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")


def _safe_float(val: object) -> float | None:
    try:
        f = float(val)  # type: ignore[arg-type]
        return round(f, 2) if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None


def fetch_betfair_historical() -> None:
    """Fetch BSP data for all race dates and match to our runners."""
    config = load_config()
    parquet_dir = PROJECT_ROOT / config["paths"]["staged_parquet"]
    output_path = parquet_dir / "betfair_historical.parquet"

    races = pd.read_parquet(parquet_dir / "races.parquet")
    runners = pd.read_parquet(parquet_dir / "runners.parquet")

    # Build lookup: (date, off_time) → list of runners
    runner_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in runners.groupby(["date", "off_time"]):
        date_val, off_val = key  # type: ignore[misc]
        runner_lookup[(str(date_val), str(off_val))] = group

    # Unique (date, course) pairs → region
    race_dates: list[tuple[datetime, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in races.iterrows():
        course = str(row["course"])
        date_str = str(row["date"])
        region = COURSE_TO_REGION.get(course)
        if region and (date_str, region) not in seen:
            seen.add((date_str, region))
            race_dates.append((datetime.strptime(date_str, "%Y-%m-%d"), region))

    race_dates.sort()
    logger.info("Fetching BSP data for %d date/region combinations", len(race_dates))

    session = requests.Session()
    matched_rows: list[dict[str, object]] = []
    total_bsp = 0
    total_matched = 0

    for i, (date, region) in enumerate(race_dates):
        csv_df = _fetch_csv(region, date, session)
        time.sleep(1)  # rate limit

        if csv_df is None or csv_df.empty:
            continue

        logger.info(
            "[%d/%d] %s %s: %d BSP rows",
            i + 1, len(race_dates), date.strftime("%Y-%m-%d"), region, len(csv_df),
        )
        total_bsp += len(csv_df)

        # Group BSP rows by (date, off_time) for matching
        for _, bsp_row in csv_df.iterrows():
            event_dt = str(bsp_row.get("EVENT_DT", ""))
            if not event_dt.strip():
                continue

            try:
                bsp_date, bsp_off = _parse_event_dt(event_dt)
            except ValueError:
                continue

            key = (bsp_date, bsp_off)
            if key not in runner_lookup:
                continue

            race_runners = runner_lookup[key]
            bsp_name = str(bsp_row.get("SELECTION_NAME", ""))
            if not bsp_name.strip():
                continue

            candidate_names = race_runners["horse_name"].tolist()
            match_name, _ratio = _best_match(bsp_name, candidate_names)
            if match_name is None:
                continue

            matched_runner = race_runners[race_runners["horse_name"] == match_name].iloc[0]
            total_matched += 1

            matched_rows.append({
                "race_id": matched_runner["race_id"],
                "date": bsp_date,
                "course": matched_runner["course"],
                "off_time": bsp_off,
                "horse_id": matched_runner["horse_id"],
                "horse_name": match_name,
                "bsp": _safe_float(bsp_row.get("BSP")),
                "wap": _safe_float(bsp_row.get("PPWAP")),
                "morning_wap": _safe_float(bsp_row.get("MORNINGWAP")),
                "pre_min": _safe_float(bsp_row.get("PPMIN")),
                "pre_max": _safe_float(bsp_row.get("PPMAX")),
                "ip_min": _safe_float(bsp_row.get("IPMIN")),
                "ip_max": _safe_float(bsp_row.get("IPMAX")),
                "morning_vol": _safe_float(bsp_row.get("MORNINGTRADEDVOL")),
                "pre_vol": _safe_float(bsp_row.get("PPTRADEDVOL")),
                "ip_vol": _safe_float(bsp_row.get("IPTRADEDVOL")),
            })

    if not matched_rows:
        logger.warning("No BSP data matched to runners")
        return

    betfair_df = pd.DataFrame(matched_rows)
    betfair_df.to_parquet(output_path, index=False)
    match_rate = total_matched / total_bsp * 100 if total_bsp > 0 else 0
    logger.info(
        "Written betfair_historical.parquet: %d rows (matched %d/%d = %.1f%%)",
        len(betfair_df), total_matched, total_bsp, match_rate,
    )
