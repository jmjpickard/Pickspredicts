"""G6: LLM-based comment extraction — extracts structured signals from race comments.

Sends runner comment text to Claude Haiku for structured extraction.
Cached in data/marts/comment_features.parquet keyed on (race_id, horse_id).
Safe to interrupt and resume (checks cache before processing).

Usage:
    python -m src.pipeline --step parse-comments
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    import anthropic as anthropic_mod  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """You are a horse racing comment analyst. Extract structured signals from race comments.
Return a JSON object with exactly these fields:
- running_style: one of (front-runner, prominent, mid-division, held-up, rear)
- finishing_effort: one of (strong, stayed-on, one-paced, weakened, pulled-up)
- trouble_in_running: true or false
- jumping: one of (fluent, adequate, mistakes, fell, unseated)
- stamina_signal: one of (stayed-on-well, found-nil, no-signal)

If the comment doesn't clearly indicate a value, use the most neutral option.
Respond with ONLY the JSON object, no other text."""


def load_config() -> dict:  # type: ignore[type-arg]
    with open(PROJECT_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def parse_comments(batch_size: int = 100) -> None:
    """Parse race comments using Claude Haiku and cache results."""
    import os

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set. Add it to .env")
        return

    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.error("anthropic package not installed. Run: uv add anthropic")
        return

    config = load_config()
    parquet_dir = PROJECT_ROOT / config["paths"]["staged_parquet"]
    marts_dir = PROJECT_ROOT / config["paths"]["marts"]
    marts_dir.mkdir(parents=True, exist_ok=True)

    cache_path = marts_dir / "comment_features.parquet"

    # Load runners with comments
    runners = pd.read_parquet(parquet_dir / "runners.parquet")
    has_comment: pd.DataFrame = runners[
        runners["comment"].notna() & (runners["comment"] != "")
    ].copy()  # type: ignore[assignment]
    logger.info("Runners with comments: %d", len(has_comment))

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_keys = set(zip(cached["race_id"], cached["horse_id"]))
        logger.info("Cached comment features: %d", len(cached))
    else:
        cached = pd.DataFrame()
        cached_keys: set[tuple[object, object]] = set()

    # Filter to uncached
    to_process: pd.DataFrame = has_comment[
        ~has_comment.apply(
            lambda r: (r["race_id"], r["horse_id"]) in cached_keys, axis=1
        )
    ].copy()  # type: ignore[assignment]
    logger.info("Comments to process: %d", len(to_process))

    if to_process.empty:
        logger.info("All comments already cached")
        return

    client = anthropic.Anthropic()
    results: list[dict[str, object]] = []

    for i, (_, row) in enumerate(to_process.iterrows()):
        if i > 0 and i % batch_size == 0:
            _save_batch(results, cached, cache_path)
            logger.info("Processed %d / %d comments", i, len(to_process))

        extracted = _extract_comment(client, str(row["comment"]))
        if extracted:
            extracted["race_id"] = row["race_id"]
            extracted["horse_id"] = row["horse_id"]
            results.append(extracted)

    _save_batch(results, cached, cache_path)
    logger.info("Comment parsing complete. Total cached: %d", len(cached) + len(results))

    # Compute aggregated features over last N runs
    _compute_derived_features(cache_path, parquet_dir / "runners.parquet", marts_dir)


def _extract_comment(
    client: anthropic_mod.Anthropic, comment: str,
) -> dict[str, object] | None:
    """Extract structured features from a single comment using Claude Haiku."""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": comment}],
        )
        text = response.content[0].text  # type: ignore[union-attr]
        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
            stripped = stripped.rsplit("```", 1)[0].strip()
        return json.loads(stripped)
    except Exception as e:
        logger.warning("Failed to parse comment: %s", e)
        return None


def _save_batch(
    results: list[dict[str, object]],
    cached: pd.DataFrame,
    cache_path: Path,
) -> None:
    """Append new results to the cache file."""
    if not results:
        return
    new_df = pd.DataFrame(results)
    combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    combined.to_parquet(cache_path, index=False)


def _compute_derived_features(
    cache_path: Path,
    runners_path: Path,
    marts_dir: Path,
) -> None:
    """Compute aggregated comment features: dominant_style, pct_trouble, pct_jumping_issues."""
    import duckdb

    comments = pd.read_parquet(cache_path)
    runners = pd.read_parquet(runners_path)

    con = duckdb.connect()
    con.register("comments", comments)
    con.register("runners", runners)

    derived = con.sql("""
        WITH ordered AS (
            SELECT
                c.*,
                CAST(r.date AS DATE) AS race_date,
                ROW_NUMBER() OVER (
                    PARTITION BY r.horse_id ORDER BY r.date, r.race_id
                ) AS run_seq
            FROM comments c
            JOIN runners r ON c.race_id = r.race_id AND c.horse_id = r.horse_id
        )
        SELECT
            t.race_id,
            t.horse_id,
            -- dominant style from prior runs
            (SELECT h.running_style
             FROM ordered h
             JOIN runners r2 ON h.race_id = r2.race_id AND h.horse_id = r2.horse_id
             WHERE r2.horse_id = r.horse_id
               AND CAST(r2.date AS DATE) < CAST(r.date AS DATE)
             GROUP BY h.running_style
             ORDER BY COUNT(*) DESC
             LIMIT 1
            ) AS dominant_style,
            -- % trouble in running over career
            (SELECT AVG(CASE WHEN h.trouble_in_running = 'true' THEN 1.0 ELSE 0 END)
             FROM ordered h
             JOIN runners r2 ON h.race_id = r2.race_id AND h.horse_id = r2.horse_id
             WHERE r2.horse_id = r.horse_id
               AND CAST(r2.date AS DATE) < CAST(r.date AS DATE)
            ) AS pct_trouble,
            -- % jumping issues over career
            (SELECT AVG(CASE WHEN h.jumping IN ('mistakes', 'fell', 'unseated')
                        THEN 1.0 ELSE 0 END)
             FROM ordered h
             JOIN runners r2 ON h.race_id = r2.race_id AND h.horse_id = r2.horse_id
             WHERE r2.horse_id = r.horse_id
               AND CAST(r2.date AS DATE) < CAST(r.date AS DATE)
            ) AS pct_jumping_issues
        FROM runners r
        JOIN comments t ON r.race_id = t.race_id AND r.horse_id = t.horse_id
    """).df()

    con.close()

    derived.to_parquet(marts_dir / "comment_derived_features.parquet", index=False)
    logger.info("Written comment_derived_features.parquet (%d rows)", len(derived))
