"""Racecard fetch health + freshness guards for scoring safety."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

FETCH_STATUS_FILENAME = "_fetch_status.yaml"
DEFAULT_MAX_AGE_MINUTES = 180


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer. Got: {value!r}") from exc


def _to_iso_date(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty ISO date string.")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD). Got: {value!r}") from exc


def get_configured_racecard_dates(config: dict[str, Any]) -> list[str] | None:
    racecards_cfg = config.get("racecards", {})
    raw_dates = racecards_cfg.get("dates")
    if raw_dates in (None, "", []):
        return None

    values: list[Any]
    if isinstance(raw_dates, str):
        values = [token for token in raw_dates.replace(",", " ").split() if token.strip()]
    elif isinstance(raw_dates, (list, tuple)):
        values = list(raw_dates)
    else:
        raise ValueError(
            f"racecards.dates must be a list of ISO dates or a whitespace/comma-separated string. "
            f"Got: {raw_dates!r}"
        )

    if not values:
        raise ValueError("racecards.dates must contain at least one ISO date.")

    dates = [
        _to_iso_date(value, name=f"racecards.dates[{idx}]")
        for idx, value in enumerate(values)
    ]
    return list(dict.fromkeys(dates))


def get_racecard_days(config: dict[str, Any]) -> int:
    days = _to_int(config["racecards"]["days"], name="racecards.days")
    if not (1 <= days <= 2):
        raise ValueError(
            f"racecards.days must be between 1 and 2 for rpscrape racecards.py. Got: {days}"
        )
    return days


def get_requested_racecard_dates(config: dict[str, Any]) -> list[str]:
    configured_dates = get_configured_racecard_dates(config)
    if configured_dates:
        return configured_dates

    days = get_racecard_days(config)
    return [(date.today() + timedelta(days=i)).isoformat() for i in range(days)]


def get_racecard_max_age_minutes(config: dict[str, Any]) -> int:
    racecards_cfg = config.get("racecards", {})
    max_age = racecards_cfg.get("max_age_minutes", DEFAULT_MAX_AGE_MINUTES)
    max_age_mins = _to_int(max_age, name="racecards.max_age_minutes")
    if max_age_mins < 1:
        raise ValueError(
            f"racecards.max_age_minutes must be >= 1. Got: {max_age_mins}"
        )
    return max_age_mins


def fetch_status_path(racecard_dir: Path) -> Path:
    return racecard_dir / FETCH_STATUS_FILENAME


def write_fetch_status(
    racecard_dir: Path,
    *,
    status: str,
    requested_dates: list[str],
    region: str | None,
    files: list[Path] | None = None,
    error: str | None = None,
) -> None:
    racecard_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": status,
        "updated_at_utc": _utc_now_iso(),
        "days": len(requested_dates),
        "requested_dates": requested_dates,
        "region": region,
    }
    if files is not None:
        payload["files"] = [p.name for p in files]
    if error:
        payload["error"] = error
    with open(fetch_status_path(racecard_dir), "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def _load_fetch_status(racecard_dir: Path) -> dict[str, Any]:
    status_file = fetch_status_path(racecard_dir)
    if not status_file.exists():
        raise RuntimeError(
            f"Racecard fetch status file missing ({status_file}). "
            "Run --step fetch-racecards before features/predict."
        )
    with open(status_file) as f:
        status = yaml.safe_load(f) or {}
    if not isinstance(status, dict):
        raise RuntimeError(
            f"Racecard fetch status file is malformed ({status_file}). "
            "Rerun --step fetch-racecards."
        )
    return status


def _latest_file_age_minutes(files: list[Path]) -> float:
    newest_mtime = max(path.stat().st_mtime for path in files)
    newest_dt = datetime.fromtimestamp(newest_mtime, tz=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - newest_dt).total_seconds()
    return age_seconds / 60.0


def validate_racecard_files(racecard_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Fail-fast guard for scoring steps that depend on fresh racecard JSON."""
    status = _load_fetch_status(racecard_dir)
    state = str(status.get("status", "")).strip().lower()
    if state != "success":
        detail = status.get("error")
        detail_text = f": {detail}" if detail else ""
        raise RuntimeError(
            f"Latest racecard fetch status is '{state or 'unknown'}'{detail_text}. "
            "Rerun --step fetch-racecards before features/predict."
        )

    json_files = sorted(racecard_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(
            f"No racecard JSON files found in {racecard_dir}. "
            "Rerun --step fetch-racecards."
        )

    status_files = status.get("files")
    if isinstance(status_files, list) and status_files:
        missing_files = [
            name for name in status_files
            if not (racecard_dir / str(name)).exists()
        ]
        if missing_files:
            raise RuntimeError(
                "Racecard fetch status references missing JSON files: "
                + ", ".join(sorted(missing_files))
                + ". Rerun --step fetch-racecards."
            )

    configured_dates = get_configured_racecard_dates(config)
    if configured_dates:
        expected_filenames = {f"{date_str}.json" for date_str in configured_dates}
        actual_filenames = {path.name for path in json_files}
        if actual_filenames != expected_filenames:
            missing = sorted(expected_filenames - actual_filenames)
            extra = sorted(actual_filenames - expected_filenames)
            detail_parts: list[str] = []
            if missing:
                detail_parts.append("missing=" + ", ".join(missing))
            if extra:
                detail_parts.append("extra=" + ", ".join(extra))
            detail = "; ".join(detail_parts)
            raise RuntimeError(
                "Racecard JSON files do not match configured racecards.dates"
                + (f" ({detail})" if detail else "")
                + ". Rerun --step fetch-racecards."
            )

    max_age_minutes = get_racecard_max_age_minutes(config)
    age_minutes = _latest_file_age_minutes(json_files)
    if age_minutes > max_age_minutes:
        raise RuntimeError(
            f"Racecard files are stale: newest JSON is {age_minutes:.1f} minutes old "
            f"(threshold={max_age_minutes} minutes). "
            "Rerun --step fetch-racecards before features/predict."
        )

    return json_files
