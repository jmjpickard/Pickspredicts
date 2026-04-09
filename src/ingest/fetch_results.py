"""Fetch historical results via rpscrape CLI.

Shells out to rpscrape.py for each configured course/year range,
then copies the CSV output to data/raw/results/.
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


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def rpscrape_output_path(course_id: int, years: str) -> Path:
    """Predict where rpscrape writes its CSV output."""
    rpscrape_data = PROJECT_ROOT / "rpscrape" / "data" / "course" / str(course_id) / "jumps"

    # rpscrape names the file based on year range: "2020_2025.csv" or "2020.csv"
    parts = years.split("-")
    if len(parts) == 2:
        filename = f"{parts[0]}_{parts[1]}.csv"
    else:
        filename = f"{parts[0]}.csv"

    return rpscrape_data / filename


def rpscrape_date_output_path(course_id: int, dates: list[str], race_type: str) -> Path:
    """Predict where rpscrape writes its date-scraped CSV output."""
    rpscrape_data = PROJECT_ROOT / "rpscrape" / "data" / "course" / str(course_id) / race_type
    sorted_dates = sorted({date_str.replace("-", "_") for date_str in dates if date_str})
    if not sorted_dates:
        raise ValueError("dates must contain at least one ISO date")

    filename = sorted_dates[0]
    if len(sorted_dates) > 1:
        filename = f"{sorted_dates[0]}_{sorted_dates[-1]}"

    return rpscrape_data / f"{filename}.csv"


def fetch_course(
    course_name: str,
    course_id: int,
    years: str,
    race_type: str,
    output_dir: Path,
    refresh: bool = False,
) -> Path | None:
    """Fetch results for a single course via rpscrape.

    Returns the path to the output CSV in data/raw/results/, or None on failure.
    """
    dest = output_dir / f"{course_name}_{years.replace('-', '_')}.csv"

    if dest.exists() and not refresh:
        row_count = sum(1 for _ in open(dest)) - 1  # subtract header
        logger.info("Skipping %s — already cached (%d rows)", course_name, row_count)
        return dest
    if dest.exists() and refresh:
        logger.info("Refreshing %s cache at %s", course_name, dest)

    logger.info("Fetching %s (course %d) for %s...", course_name, course_id, years)

    cmd = [
        sys.executable,
        "rpscrape.py",
        "-c", str(course_id),
        "-y", years,
        "-t", race_type,
    ]

    # Stream stdout to terminal in real-time, capture lines for OUTPUT_CSV parsing
    captured_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(RPSCRAPE_SCRIPTS),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        captured_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        logger.error("rpscrape failed for %s:\n%s", course_name, stderr)
        return None

    # Find the output CSV path from rpscrape's stdout
    output_csv: Path | None = None
    for line in captured_lines:
        if line.startswith("OUTPUT_CSV="):
            output_csv = Path(line.split("=", 1)[1])
            break

    # Fall back to predicted path
    if output_csv is None or not output_csv.exists():
        output_csv = rpscrape_output_path(course_id, years)

    if not output_csv.exists():
        logger.error("Could not find rpscrape output for %s", course_name)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_csv, dest)

    row_count = sum(1 for _ in open(dest)) - 1
    logger.info("Fetched %s — %d rows → %s", course_name, row_count, dest)
    return dest


def fetch_course_dates(
    course_name: str,
    course_id: int,
    dates: list[str],
    race_type: str,
    output_dir: Path,
    refresh: bool = False,
) -> Path | None:
    """Fetch results for a single course across explicit dates via rpscrape."""
    sorted_dates = sorted({str(date_str).strip() for date_str in dates if str(date_str).strip()})
    if not sorted_dates:
        raise ValueError("dates must contain at least one ISO date")

    years = sorted({date_str[:4] for date_str in sorted_dates if len(date_str) >= 4})
    if not years:
        raise ValueError("dates must include at least one ISO year")

    filename_years = years[0] if len(years) == 1 else f"{years[0]}_{years[-1]}"
    dest = output_dir / f"{course_name}_{filename_years}.csv"

    if dest.exists() and not refresh:
        row_count = sum(1 for _ in open(dest)) - 1
        logger.info("Skipping %s date refresh — already cached (%d rows)", course_name, row_count)
        return dest
    if dest.exists() and refresh:
        logger.info("Refreshing %s date cache at %s", course_name, dest)

    date_tokens = [date_str.replace("-", "/") for date_str in sorted_dates]
    if len(date_tokens) == 1:
        date_arg = date_tokens[0]
    else:
        date_arg = f"{date_tokens[0]}-{date_tokens[-1]}"

    logger.info(
        "Fetching %s (course %d) for date range %s...",
        course_name,
        course_id,
        date_arg,
    )

    cmd = [
        sys.executable,
        "rpscrape.py",
        "-c",
        str(course_id),
        "-d",
        date_arg,
        "-t",
        race_type,
        "--clean",
    ]

    captured_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(RPSCRAPE_SCRIPTS),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        captured_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        logger.error("rpscrape date fetch failed for %s:\n%s", course_name, stderr)
        return None

    output_csv: Path | None = None
    for line in captured_lines:
        if line.startswith("OUTPUT_CSV="):
            output_csv = Path(line.split("=", 1)[1])
            break

    if output_csv is None or not output_csv.exists():
        output_csv = rpscrape_date_output_path(course_id, sorted_dates, race_type)

    if not output_csv.exists():
        logger.error("Could not find rpscrape date output for %s", course_name)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_csv, dest)

    row_count = sum(1 for _ in open(dest)) - 1
    logger.info("Fetched %s by date — %d rows → %s", course_name, row_count, dest)
    return dest


def fetch_all_results() -> list[Path]:
    """Fetch results for all configured courses. Returns list of output CSV paths."""
    config = load_config()
    output_dir = PROJECT_ROOT / config["paths"]["raw_results"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []

    for course_name, course_config in config["courses"].items():
        course_id = course_config["id"]
        if course_id is None:
            logger.warning("Skipping %s — no course ID configured", course_name)
            continue

        path = fetch_course(
            course_name=course_name,
            course_id=course_id,
            years=course_config["years"],
            race_type=course_config["type"],
            output_dir=output_dir,
        )
        if path is not None:
            results.append(path)

    logger.info("Fetched %d/%d courses", len(results), len(config["courses"]))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetch_all_results()
