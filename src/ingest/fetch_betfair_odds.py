"""Betfair Exchange API client — cert-based auth, market discovery, live pricing.

Fetches real-time pre-race exchange odds for Cheltenham 2026 win markets.
Outputs data in the same schema as betfair_historical.parquet where possible.

Setup:
1. Generate self-signed cert: openssl req -x509 -nodes -newkey rsa:2048 -keyout configs/certs/client.key -out configs/certs/client.crt -days 365
2. Upload .crt to https://developer.betfair.com/ → My Security
3. Set env vars: BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD

Usage:
    python -m src.pipeline --step fetch-odds
"""

import logging
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from src.ingest.racecard_health import get_requested_racecard_dates

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ = load_dotenv(PROJECT_ROOT / ".env")

CERT_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
API_BASE = "https://api.betfair.com/exchange/betting/rest/v1.0"
MARKET_BOOK_BATCH_SIZE = 5

# Betfair event type ID for horse racing
HORSE_RACING_EVENT_TYPE = "7"


def load_config() -> dict:  # type: ignore[type-arg]
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_coverage_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info("Written odds coverage report: %s", report_path)


def _clean_name(name: str) -> str:
    """Normalize runner names for fuzzy matching."""
    cleaned = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.strip().lower())
    return cleaned.strip()


def _normalise_off_time(value: object) -> str:
    text = str(value).strip()
    return text[:5] if len(text) >= 5 else text


def _off_time_minutes(value: object) -> int | None:
    text = _normalise_off_time(value)
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return None


def _best_match(
    source_name: str, candidates: list[str], threshold: float,
) -> tuple[str | None, float]:
    best_name: str | None = None
    best_ratio = 0.0
    clean_source = _clean_name(source_name)

    for candidate in candidates:
        ratio = SequenceMatcher(None, clean_source, _clean_name(candidate)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = candidate

    if best_ratio >= threshold:
        return best_name, best_ratio
    return None, best_ratio


def _clean_race_title(name: str) -> str:
    cleaned = re.sub(r"\(.*?\)", " ", name.lower())
    cleaned = cleaned.replace("novices'", "novice")
    cleaned = cleaned.replace("novices", "novice")
    cleaned = cleaned.replace("challenge trophy", " ")
    cleaned = cleaned.replace("trophy", " ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    stopwords = {"grade", "listed", "registered", "as", "gbb", "race", "the"}
    tokens = [token for token in cleaned.split() if token and token not in stopwords]
    return " ".join(tokens)


def _best_race_match(
    source_name: str, candidates: list[str], threshold: float,
) -> tuple[str | None, float]:
    best_name: str | None = None
    best_ratio = 0.0
    clean_source = _clean_race_title(source_name)

    for candidate in candidates:
        ratio = SequenceMatcher(None, clean_source, _clean_race_title(candidate)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = candidate

    if best_ratio >= threshold:
        return best_name, best_ratio
    return None, best_ratio


def _load_racecard_reference(config: dict[str, Any]) -> pd.DataFrame:
    """Load racecard runners used to map Betfair IDs onto RP IDs."""
    racecard_parquet = PROJECT_ROOT / config["paths"]["marts"] / "racecard_runners.parquet"
    racecard_dir = PROJECT_ROOT / config["paths"]["raw_racecards"]
    raw_json_files = sorted(racecard_dir.glob("*.json")) if racecard_dir.exists() else []
    latest_raw_mtime = max((path.stat().st_mtime for path in raw_json_files), default=None)

    if racecard_parquet.exists():
        parquet_is_fresh = (
            latest_raw_mtime is None or racecard_parquet.stat().st_mtime >= latest_raw_mtime
        )
        if parquet_is_fresh:
            reference = pd.read_parquet(racecard_parquet)
            logger.info(
                "Loaded racecard reference from %s (%d rows)",
                racecard_parquet,
                len(reference),
            )
            return reference

        logger.info(
            "Racecard reference cache is older than raw racecards; rebuilding %s",
            racecard_parquet,
        )

    try:
        from src.features.racecard import load_racecards
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to import racecard loader for ID mapping: %s", exc)
        return pd.DataFrame()

    reference = load_racecards()
    if reference.empty:
        logger.warning(
            "No racecard runners available for Betfair ID mapping. "
            "Run --step fetch-racecards (and optionally --step features) first."
        )
        return reference

    reference.to_parquet(racecard_parquet, index=False)
    logger.info("Cached racecard reference at %s (%d rows)", racecard_parquet, len(reference))
    return reference


def _scoring_courses(config: dict[str, Any]) -> list[str]:
    env_override = os.getenv("SCORING_COURSES", "").strip()
    if env_override:
        courses = [token.strip() for token in env_override.split(",") if token.strip()]
        if courses:
            return courses

    racecards_cfg = config.get("racecards", {})
    configured = racecards_cfg.get("scoring_courses")
    if isinstance(configured, list) and configured:
        return [str(c).strip() for c in configured if str(c).strip()]

    return ["Cheltenham"]


def _same_course(a: str, b: str) -> bool:
    a_norm = re.sub(r"\s*\(.*?\)\s*", " ", a).strip().lower()
    b_norm = re.sub(r"\s*\(.*?\)\s*", " ", b).strip().lower()
    return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm


def _market_type_code(market: dict[str, Any]) -> str:
    description = market.get("description", {})
    if not isinstance(description, dict):
        return ""
    return str(description.get("marketType", "")).strip().upper()


def _market_type_rank(code: str) -> int:
    if code == "WIN":
        return 0
    if code == "ANTEPOST_WIN":
        return 1
    return 2


def _map_exchange_to_rp_ids(
    exchange_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map exchange market/selection IDs onto RP race_id/horse_id using racecards."""
    mapping_stats: dict[str, Any] = {
        "input_rows": int(len(exchange_df)),
        "mapped_rows": 0,
        "unmatched_rows": 0,
        "mapped_ratio": 0.0,
        "unmatched_by_reason": {},
    }
    if exchange_df.empty or reference_df.empty:
        mapping_stats["unmatched_rows"] = int(len(exchange_df))
        return exchange_df, mapping_stats

    required_cols = {"race_id", "horse_id", "horse_name", "date", "off_time", "course"}
    if not required_cols.issubset(reference_df.columns):
        logger.warning("Racecard reference missing required ID columns; skipping mapping")
        mapping_stats["unmatched_rows"] = int(len(exchange_df))
        mapping_stats["unmatched_by_reason"] = {"missing_reference_columns": int(len(exchange_df))}
        return exchange_df, mapping_stats

    reference = reference_df.copy()
    reference["date"] = reference["date"].astype(str).str[:10]
    reference["off_time"] = reference["off_time"].apply(_normalise_off_time)
    reference["horse_name"] = reference["horse_name"].astype(str)
    reference["course"] = reference["course"].astype(str)
    if "race_name" in reference.columns:
        reference["race_name"] = reference["race_name"].astype(str)
    reference = reference.dropna(
        subset=["race_id", "horse_id", "horse_name", "date", "off_time", "course"]
    )
    reference = reference.drop_duplicates(subset=["race_id", "horse_id"])

    mapped_rows: list[dict[str, object]] = []
    unmatched_reasons: dict[str, int] = {
        "missing_date_bucket": 0,
        "missing_course_bucket": 0,
        "race_name_below_threshold": 0,
        "time_bucket_miss": 0,
        "name_below_threshold": 0,
    }
    race_name_threshold = max(0.45, min(threshold, 0.6))

    for _, row in exchange_df.iterrows():
        date_key = str(row.get("date", ""))[:10]
        if not date_key:
            unmatched_reasons["missing_date_bucket"] += 1
            continue

        candidates = reference[reference["date"] == date_key]
        if candidates.empty:
            unmatched_reasons["missing_date_bucket"] += 1
            continue

        course_name = str(row.get("course", "")).strip()
        if course_name:
            candidates = candidates[candidates["course"].apply(lambda value: _same_course(value, course_name))]
        if candidates.empty:
            unmatched_reasons["missing_course_bucket"] += 1
            continue

        race_candidates = candidates
        market_name = str(row.get("market_name", "")).strip()
        if market_name and "race_name" in candidates.columns:
            race_lookup = candidates[["race_id", "race_name"]].drop_duplicates()
            match_race_name, _ = _best_race_match(
                market_name,
                race_lookup["race_name"].tolist(),
                threshold=race_name_threshold,
            )
            if match_race_name is not None:
                matched_race_ids = race_lookup.loc[
                    race_lookup["race_name"] == match_race_name,
                    "race_id",
                ].astype(str)
                race_candidates = candidates[
                    candidates["race_id"].astype(str).isin(set(matched_race_ids.tolist()))
                ]
            else:
                unmatched_reasons["race_name_below_threshold"] += 1
                race_candidates = candidates.iloc[0:0]

        if race_candidates.empty:
            off_key = _normalise_off_time(row.get("off_time", ""))
            target_minutes = _off_time_minutes(off_key)
            if target_minutes is None:
                unmatched_reasons["time_bucket_miss"] += 1
                continue

            race_times = candidates[["race_id", "off_time"]].drop_duplicates()
            race_times["off_minutes"] = race_times["off_time"].apply(_off_time_minutes)
            race_times = race_times.dropna(subset=["off_minutes"])
            if race_times.empty:
                unmatched_reasons["time_bucket_miss"] += 1
                continue

            race_times["time_diff"] = (race_times["off_minutes"] - target_minutes).abs()
            nearest = race_times.sort_values("time_diff").iloc[0]
            if int(nearest["time_diff"]) > 15:
                unmatched_reasons["time_bucket_miss"] += 1
                continue
            race_candidates = candidates[
                candidates["race_id"].astype(str) == str(nearest["race_id"])
            ]

        candidate_names = race_candidates["horse_name"].tolist()
        match_name, score = _best_match(
            str(row.get("horse_name", "")),
            candidate_names,
            threshold=threshold,
        )
        if match_name is None:
            unmatched_reasons["name_below_threshold"] += 1
            continue

        matched_row = race_candidates[race_candidates["horse_name"] == match_name].iloc[0]
        mapped = row.to_dict()
        mapped["betfair_market_id"] = str(mapped["race_id"])
        betfair_selection = mapped.get("horse_id")
        mapped["betfair_selection_id"] = (
            int(betfair_selection) if pd.notna(betfair_selection) else None
        )
        mapped["race_id"] = str(matched_row["race_id"])
        mapped["horse_id"] = int(matched_row["horse_id"])
        mapped["horse_name"] = str(matched_row["horse_name"])
        mapped["mapping_status"] = "mapped"
        mapped["match_score"] = round(float(score), 4)
        mapped_rows.append(mapped)

    if not mapped_rows:
        logger.warning("Betfair ID mapping found 0 matches; writing raw IDs")
        raw_df = exchange_df.copy()
        raw_df["mapping_status"] = "raw_unmapped"
        raw_df["match_score"] = None
        mapping_stats["unmatched_rows"] = int(len(exchange_df))
        mapping_stats["unmatched_by_reason"] = unmatched_reasons
        return raw_df, mapping_stats

    mapped_df = pd.DataFrame(mapped_rows)
    # Prefer rows with actionable live prices when multiple markets map to the same runner.
    price_rank = {
        "best_back": 2,
        "last_traded": 1,
        "none": 0,
    }
    price_source = (
        mapped_df["price_source"]
        if "price_source" in mapped_df.columns
        else pd.Series("none", index=mapped_df.index)
    )
    market_status = (
        mapped_df["market_status"]
        if "market_status" in mapped_df.columns
        else pd.Series("", index=mapped_df.index)
    )
    market_type = (
        mapped_df["market_type"]
        if "market_type" in mapped_df.columns
        else pd.Series("", index=mapped_df.index)
    )
    mapped_df["_price_rank"] = price_source.map(price_rank).fillna(0).astype(int)
    mapped_df["_status_rank"] = market_status.astype(str).str.upper().eq("OPEN").astype(int)
    mapped_df["_type_rank"] = (
        market_type.astype(str).str.upper().map({"WIN": 2, "ANTEPOST_WIN": 1}).fillna(0).astype(int)
    )
    mapped_df = mapped_df.sort_values(
        by=["race_id", "horse_id", "_price_rank", "_status_rank", "_type_rank", "match_score"],
        ascending=[True, True, False, False, False, False],
    )
    mapped_df = mapped_df.drop_duplicates(subset=["race_id", "horse_id"], keep="first")
    mapped_df = mapped_df.drop(columns=["_price_rank", "_status_rank", "_type_rank"], errors="ignore")
    mapping_stats["mapped_rows"] = int(len(mapped_df))
    mapping_stats["unmatched_rows"] = int(max(len(exchange_df) - len(mapped_df), 0))
    mapping_stats["mapped_ratio"] = round(float(len(mapped_df) / len(exchange_df)), 4)
    mapping_stats["unmatched_by_reason"] = unmatched_reasons

    logger.info(
        "Mapped Betfair IDs to RP IDs: %d/%d matched (%.1f%%), %d unmatched",
        len(mapped_df),
        len(exchange_df),
        100 * (len(mapped_df) / len(exchange_df)),
        mapping_stats["unmatched_rows"],
    )
    return mapped_df, mapping_stats


class BetfairClient:
    """Betfair Exchange API-NG client with cert-based login."""

    def __init__(self, app_key: str, username: str, password: str) -> None:
        self.app_key = app_key
        self.username = username
        self.password = password
        self._session_token: str | None = None
        self._session = requests.Session()

    def login(self) -> None:
        """Authenticate via cert-based SSO login."""
        cert_dir = PROJECT_ROOT / "configs" / "certs"
        cert_path = cert_dir / "client.crt"
        key_path = cert_dir / "client.key"

        if not cert_path.exists() or not key_path.exists():
            raise FileNotFoundError(
                f"Betfair certs not found at {cert_dir}. "
                "Generate with: openssl req -x509 -nodes -newkey rsa:2048 "
                "-keyout configs/certs/client.key -out configs/certs/client.crt -days 365"
            )

        resp = self._session.post(
            CERT_LOGIN_URL,
            data={"username": self.username, "password": self.password},
            cert=(str(cert_path), str(key_path)),
            headers={"X-Application": self.app_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("loginStatus") != "SUCCESS":
            raise RuntimeError(f"Betfair login failed: {data.get('loginStatus')}")

        token: str = data["sessionToken"]
        self._session_token = token
        self._session.headers.update({
            "X-Application": self.app_key,
            "X-Authentication": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        logger.info("Betfair login successful")

    def _api_call(self, operation: str, params: dict[str, Any]) -> Any:
        """Make an API-NG call."""
        if not self._session_token:
            raise RuntimeError("Not logged in — call login() first")

        url = f"{API_BASE}/{operation}/"
        resp = self._session.post(url, json={"filter": params.get("filter", {}), **params}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _http_error_detail(exc: requests.RequestException) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return str(exc)
        body = response.text.strip()
        if len(body) > 500:
            body = body[:500] + "..."
        if body:
            return f"{exc} | response={body}"
        return str(exc)

    def find_markets(
        self, event_name: str, market_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find win markets for a given event (e.g. 'Cheltenham')."""
        requested_types = market_types or ["WIN", "ANTEPOST_WIN"]
        result: list[dict[str, Any]] = self._api_call("listMarketCatalogue", {
            "filter": {
                "eventTypeIds": [HORSE_RACING_EVENT_TYPE],
                "marketTypeCodes": requested_types,
                "textQuery": event_name,
            },
            "marketProjection": [
                "MARKET_START_TIME",
                "RUNNER_DESCRIPTION",
                "EVENT",
                "MARKET_DESCRIPTION",
            ],
            "maxResults": "200",
            "sort": "FIRST_TO_START",
        })
        return result

    def get_prices(self, market_ids: list[str]) -> list[dict[str, Any]]:
        """Get current prices for a list of market IDs."""
        books: list[dict[str, Any]] = []
        for start in range(0, len(market_ids), MARKET_BOOK_BATCH_SIZE):
            batch = market_ids[start:start + MARKET_BOOK_BATCH_SIZE]
            try:
                batch_result: list[dict[str, Any]] = self._api_call("listMarketBook", {
                    "marketIds": batch,
                    "priceProjection": {"priceData": ["EX_BEST_OFFERS", "EX_TRADED"]},
                })
                books.extend(batch_result)
                continue
            except requests.RequestException as exc:
                logger.warning(
                    "listMarketBook batch failed for %d markets: %s",
                    len(batch),
                    self._http_error_detail(exc),
                )

            # Fallback: isolate any bad market IDs so healthy markets still flow through.
            for market_id in batch:
                try:
                    single_result: list[dict[str, Any]] = self._api_call("listMarketBook", {
                        "marketIds": [market_id],
                        "priceProjection": {"priceData": ["EX_BEST_OFFERS", "EX_TRADED"]},
                    })
                except requests.RequestException as inner_exc:
                    logger.warning(
                        "Skipping market %s after listMarketBook failure: %s",
                        market_id,
                        self._http_error_detail(inner_exc),
                    )
                    continue
                books.extend(single_result)
        return books


def fetch_cheltenham_odds() -> None:
    """Fetch live exchange odds for configured scoring courses and write to parquet."""
    config = load_config()
    output_dir = PROJECT_ROOT / config["paths"]["raw_betfair"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exchange_odds.parquet"
    scoring_courses = _scoring_courses(config)
    target_dates = set(get_requested_racecard_dates(config))
    fetched_at_utc = _utc_now_iso()

    coverage_report: dict[str, Any] = {
        "run_started_at_utc": fetched_at_utc,
        "status": "starting",
        "scoring_courses": scoring_courses,
    }

    app_key = os.environ.get("BETFAIR_APP_KEY", "")
    username = os.environ.get("BETFAIR_USERNAME", "")
    password = os.environ.get("BETFAIR_PASSWORD", "")

    missing_env = [
        name for name, val in {
            "BETFAIR_APP_KEY": app_key,
            "BETFAIR_USERNAME": username,
            "BETFAIR_PASSWORD": password,
        }.items() if not val
    ]
    if missing_env:
        logger.error(
            "Missing Betfair credentials. Set BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD"
        )
        coverage_report["status"] = "blocked_missing_credentials"
        coverage_report["missing_env"] = missing_env
        _write_coverage_report(output_dir, coverage_report)
        return

    cert_dir = PROJECT_ROOT / "configs" / "certs"
    cert_path = cert_dir / "client.crt"
    key_path = cert_dir / "client.key"
    if not cert_path.exists() or not key_path.exists():
        logger.error(
            "Betfair cert files missing. Expected %s and %s",
            cert_path,
            key_path,
        )
        coverage_report["status"] = "blocked_missing_certs"
        coverage_report["required_cert_paths"] = {
            "client_crt": str(cert_path),
            "client_key": str(key_path),
        }
        _write_coverage_report(output_dir, coverage_report)
        return

    client = BetfairClient(app_key, username, password)
    try:
        client.login()
    except Exception as exc:
        logger.exception("Betfair login failed")
        coverage_report["status"] = "login_failed"
        coverage_report["error"] = str(exc)
        _write_coverage_report(output_dir, coverage_report)
        return

    # Find win markets for requested courses
    markets: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    found_markets_by_course: dict[str, int] = {}
    for course in scoring_courses:
        found = client.find_markets(course)
        preferred_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for market in found:
            venue = str(market.get("event", {}).get("venue", ""))
            event_name = str(market.get("event", {}).get("name", ""))
            market_start = str(market.get("marketStartTime", ""))
            if venue:
                if not _same_course(venue, course):
                    continue
            elif event_name and course.strip().lower() not in event_name.strip().lower():
                continue

            if target_dates and market_start:
                try:
                    market_date = datetime.fromisoformat(
                        market_start.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                except ValueError:
                    market_date = ""
                if market_date and market_date not in target_dates:
                    continue

            key = (
                event_name,
                str(market.get("marketName", "")),
                market_start,
            )
            existing = preferred_by_key.get(key)
            if existing is None:
                preferred_by_key[key] = market
                continue

            existing_rank = _market_type_rank(_market_type_code(existing))
            candidate_rank = _market_type_rank(_market_type_code(market))
            if candidate_rank < existing_rank:
                preferred_by_key[key] = market

        filtered: list[dict[str, Any]] = []
        for market in preferred_by_key.values():
            market_id = str(market.get("marketId", ""))
            if not market_id or market_id in seen_market_ids:
                continue
            seen_market_ids.add(market_id)
            filtered.append(market)

        logger.info("Found %d %s win markets", len(filtered), course)
        found_markets_by_course[course] = len(filtered)
        markets.extend(filtered)

    if not markets:
        logger.warning("No markets found for configured scoring courses: %s", scoring_courses)
        coverage_report["status"] = "no_markets_found"
        coverage_report["found_markets_by_course"] = found_markets_by_course
        coverage_report["total_markets"] = 0
        _write_coverage_report(output_dir, coverage_report)
        return

    logger.info("Found %d total win markets for scoring courses", len(markets))

    # Get prices for all markets (batch by 20)
    rows: list[dict[str, object]] = []

    for i in range(0, len(markets), 20):
        batch = markets[i:i + 20]
        market_ids = [m["marketId"] for m in batch]
        market_lookup = {m["marketId"]: m for m in batch}

        books = client.get_prices(market_ids)

        for book in books:
            market_id = book["marketId"]
            catalogue = market_lookup.get(market_id, {})
            event = catalogue.get("event", {})
            market_start = catalogue.get("marketStartTime", "")
            description = catalogue.get("description", {})
            market_status = str(book.get("status", "")).upper()
            market_type = _market_type_code(catalogue)
            turn_in_play = bool(description.get("turnInPlayEnabled", False))

            # Parse start time for date/off_time
            date_str = ""
            off_time = ""
            if market_start:
                try:
                    dt = datetime.fromisoformat(market_start.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    off_time = dt.strftime("%H:%M")
                except ValueError:
                    pass

            # Build runner name lookup from catalogue
            runner_names: dict[int, str] = {}
            for runner_desc in catalogue.get("runners", []):
                runner_names[runner_desc["selectionId"]] = runner_desc.get("runnerName", "")

            for runner in book.get("runners", []):
                sel_id = runner["selectionId"]
                name = runner_names.get(sel_id, "")

                # Best back/lay
                back_prices = runner.get("ex", {}).get("availableToBack", [])
                lay_prices = runner.get("ex", {}).get("availableToLay", [])
                last_price_traded = runner.get("lastPriceTraded")
                best_back = back_prices[0]["price"] if back_prices else None
                best_lay = lay_prices[0]["price"] if lay_prices else None
                price_source = "none"
                if best_back is not None:
                    price_source = "best_back"
                elif last_price_traded is not None:
                    best_back = last_price_traded
                    price_source = "last_traded"

                # Traded volume
                traded_vol = runner.get("ex", {}).get("tradedVolume", [])
                total_vol = sum(t.get("size", 0) for t in traded_vol)
                if total_vol <= 0:
                    total_matched = runner.get("totalMatched")
                    if isinstance(total_matched, (int, float)):
                        total_vol = float(total_matched)

                # Use best back as morning_wap proxy for market features
                rows.append({
                    "race_id": market_id,
                    "date": date_str,
                    "course": event.get("venue", "Cheltenham"),
                    "off_time": off_time,
                    "market_name": catalogue.get("marketName", ""),
                    "horse_id": sel_id,
                    "horse_name": name,
                    "market_status": market_status,
                    "market_type": market_type,
                    "turn_in_play_enabled": turn_in_play,
                    "runner_status": runner.get("status"),
                    "last_price_traded": last_price_traded,
                    "price_source": price_source,
                    "bsp": None,  # not available pre-race
                    "wap": best_back,
                    "morning_wap": best_back,
                    "pre_min": best_back,
                    "pre_max": best_lay,
                    "ip_min": None,
                    "ip_max": None,
                    "morning_vol": total_vol,
                    "pre_vol": total_vol,
                    "ip_vol": None,
                    "fetched_at_utc": fetched_at_utc,
                })

    if not rows:
        logger.warning("No price data retrieved")
        coverage_report["status"] = "no_price_data"
        coverage_report["found_markets_by_course"] = found_markets_by_course
        coverage_report["total_markets"] = int(len(markets))
        _write_coverage_report(output_dir, coverage_report)
        return

    df = pd.DataFrame(rows)
    match_threshold = float(config.get("betfair", {}).get("match_threshold", 0.70))
    reference = _load_racecard_reference(config)
    df, mapping_stats = _map_exchange_to_rp_ids(df, reference, threshold=match_threshold)
    df.to_parquet(output_path, index=False)
    logger.info("Written exchange_odds.parquet: %d rows across %d markets", len(df), len(markets))

    rows_with_back_price = int((df["price_source"] == "best_back").sum()) if "price_source" in df.columns else 0
    rows_with_last_traded = int((df["price_source"] == "last_traded").sum()) if "price_source" in df.columns else 0
    rows_with_any_price = int(df["wap"].notna().sum()) if "wap" in df.columns else 0
    rows_with_lay_price = int(df["pre_max"].notna().sum()) if "pre_max" in df.columns else 0
    unique_races = int(df["race_id"].astype(str).nunique()) if "race_id" in df.columns else 0

    coverage_report["status"] = "success"
    coverage_report["finished_at_utc"] = _utc_now_iso()
    coverage_report["found_markets_by_course"] = found_markets_by_course
    coverage_report["total_markets"] = int(len(markets))
    coverage_report["rows_written"] = int(len(df))
    coverage_report["unique_races_written"] = unique_races
    coverage_report["rows_with_back_price"] = rows_with_back_price
    coverage_report["rows_with_last_traded_price"] = rows_with_last_traded
    coverage_report["rows_with_any_price"] = rows_with_any_price
    coverage_report["rows_with_lay_price"] = rows_with_lay_price
    coverage_report["rows_with_back_price_pct"] = (
        round(rows_with_back_price / len(df), 4) if len(df) else 0.0
    )
    coverage_report["rows_with_any_price_pct"] = (
        round(rows_with_any_price / len(df), 4) if len(df) else 0.0
    )
    coverage_report["mapping"] = mapping_stats
    coverage_report["match_threshold"] = match_threshold
    coverage_report["output_path"] = str(output_path)

    if not reference.empty and {"race_id", "horse_id"}.issubset(reference.columns):
        expected_runners = int(reference.drop_duplicates(subset=["race_id", "horse_id"]).shape[0])
        coverage_report["expected_runners_from_racecards"] = expected_runners
        coverage_report["runner_coverage_vs_racecards"] = (
            round(mapping_stats.get("mapped_rows", 0) / expected_runners, 4)
            if expected_runners else 0.0
        )

    _write_coverage_report(output_dir, coverage_report)
