"""Fetch upcoming racecards via rpscrape's racecards.py.

Shells out to racecards.py for configured day range and region,
then copies JSON output to data/raw/racecards/.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.ingest.racecard_health import get_requested_racecard_dates, write_fetch_status

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RPSCRAPE_SCRIPTS = PROJECT_ROOT / "rpscrape" / "scripts"
RPSCRAPE_RACECARDS = PROJECT_ROOT / "rpscrape" / "racecards"
VALID_REGION_CODES = {"gb", "ire"}


def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_region(region_raw: Any) -> str | None:
    if region_raw is None:
        return None

    tokens: list[str] = []
    if isinstance(region_raw, (list, tuple)):
        for item in region_raw:
            tokens.extend(str(item).strip().lower().split())
    else:
        tokens.extend(str(region_raw).strip().lower().replace(",", " ").split())

    if not tokens:
        return None

    invalid = [token for token in tokens if token not in VALID_REGION_CODES]
    if invalid:
        raise ValueError(
            f"racecards.region contains invalid code(s): {', '.join(sorted(set(invalid)))}. "
            "Use 'gb', 'ire', or both."
        )

    unique_tokens = list(dict.fromkeys(tokens))
    if len(unique_tokens) == 1:
        return unique_tokens[0]

    logger.info(
        "racecards.region has multiple codes (%s); fetching all regions in one run",
        " ".join(unique_tokens),
    )
    return None


def fetch_racecards() -> list[Path]:
    """Fetch racecards for upcoming days. Returns list of JSON file paths."""
    config = load_config()
    rc_config = config["racecards"]
    requested_dates = get_requested_racecard_dates(config)
    region = _resolve_region(rc_config.get("region"))

    output_dir = PROJECT_ROOT / config["paths"]["raw_racecards"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fetch_status(
        output_dir,
        status="started",
        requested_dates=requested_dates,
        region=region,
    )

    logger.info(
        "Fetching racecards for %d configured date(s): %s (region=%s)",
        len(requested_dates),
        ", ".join(requested_dates),
        region,
    )

    for date_str in requested_dates:
        cmd = [
            sys.executable,
            "racecards.py",
            "--date",
            date_str,
        ]
        if region:
            cmd.extend(["--region", region])

        result = subprocess.run(
            cmd,
            cwd=str(RPSCRAPE_SCRIPTS),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            stdout = result.stdout.strip() if result.stdout else ""
            stderr = result.stderr.strip() if result.stderr else ""
            detail_parts: list[str] = [f"date: {date_str}"]
            if stdout:
                detail_parts.append(f"stdout:\n{stdout}")
            if stderr:
                detail_parts.append(f"stderr:\n{stderr}")
            detail = "\n\n".join(detail_parts)
            logger.error("racecards.py failed (exit=%d):\n%s", result.returncode, detail)
            write_fetch_status(
                output_dir,
                status="failed",
                requested_dates=requested_dates,
                region=region,
                error=detail,
            )
            raise RuntimeError("racecards.py failed; refusing to continue with stale racecards")

    expected_filenames = {f"{d}.json" for d in requested_dates}

    # racecards.py writes JSON to rpscrape/racecards/{date}.json
    copied: list[Path] = []
    if not RPSCRAPE_RACECARDS.exists():
        msg = f"Expected racecard directory does not exist: {RPSCRAPE_RACECARDS}"
        write_fetch_status(
            output_dir,
            status="failed",
            requested_dates=requested_dates,
            region=region,
            error=msg,
        )
        raise RuntimeError(msg)

    # Prune stale destination files so scoring only sees current fetch window.
    for existing in sorted(output_dir.glob("*.json")):
        if existing.name not in expected_filenames:
            existing.unlink()
            logger.info("Removed stale racecard: %s", existing.name)

    missing_sources: list[str] = []
    for date_str in requested_dates:
        src = RPSCRAPE_RACECARDS / f"{date_str}.json"
        if not src.exists():
            missing_sources.append(src.name)
            continue
        dest = output_dir / src.name
        shutil.copy2(src, dest)
        logger.info("Copied racecard: %s", dest.name)
        copied.append(dest)

    if missing_sources:
        msg = (
            "racecards.py did not produce all expected JSON files: "
            + ", ".join(missing_sources)
        )
        write_fetch_status(
            output_dir,
            status="failed",
            requested_dates=requested_dates,
            region=region,
            error=msg,
        )
        raise RuntimeError(msg)

    if not copied:
        msg = "racecards.py completed but produced no expected JSON files"
        write_fetch_status(
            output_dir,
            status="failed",
            requested_dates=requested_dates,
            region=region,
            error=msg,
        )
        raise RuntimeError(msg)

    write_fetch_status(
        output_dir,
        status="success",
        requested_dates=requested_dates,
        region=region,
        files=copied,
    )

    logger.info("Fetched %d racecard files", len(copied))
    return copied


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetch_racecards()
