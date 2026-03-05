"""Fetch upcoming racecards via rpscrape's racecards.py.

Shells out to racecards.py for configured day range and region,
then copies JSON output to data/raw/racecards/.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RPSCRAPE_SCRIPTS = PROJECT_ROOT / "rpscrape" / "scripts"
RPSCRAPE_RACECARDS = PROJECT_ROOT / "rpscrape" / "racecards"


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_racecards() -> list[Path]:
    """Fetch racecards for upcoming days. Returns list of JSON file paths."""
    config = load_config()
    rc_config = config["racecards"]
    days = rc_config["days"]
    region = rc_config.get("region")

    output_dir = PROJECT_ROOT / config["paths"]["raw_racecards"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "racecards.py",
        "--days", str(days),
    ]
    if region:
        cmd.extend(["--region", region])

    logger.info("Fetching racecards for next %d days (region=%s)...", days, region)

    result = subprocess.run(
        cmd,
        cwd=str(RPSCRAPE_SCRIPTS),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        logger.error("racecards.py failed:\n%s", result.stderr)
        return []

    # racecards.py writes JSON to rpscrape/racecards/{date}.json
    copied: list[Path] = []
    if RPSCRAPE_RACECARDS.exists():
        for json_file in sorted(RPSCRAPE_RACECARDS.glob("*.json")):
            dest = output_dir / json_file.name
            shutil.copy2(json_file, dest)
            logger.info("Copied racecard: %s", dest.name)
            copied.append(dest)

    logger.info("Fetched %d racecard files", len(copied))
    return copied


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetch_racecards()
