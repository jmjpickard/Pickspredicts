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
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    import anthropic as anthropic_mod  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "claude-haiku-4-5-20251001"

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


def parse_comments(
    batch_size: int = 100,
    mode: str = "sync",
    poll_interval_secs: int = 15,
    max_wait_minutes: int = 120,
) -> None:
    """Parse race comments using Claude Haiku and cache results.

    Args:
        batch_size: Sync mode flush cadence; batch mode request count per API batch.
        mode: "sync" (default) or "batch" (Anthropic Message Batches API).
        poll_interval_secs: Batch mode polling interval.
        max_wait_minutes: Batch mode timeout per submitted batch.
    """
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

    mode = mode.strip().lower()
    if mode not in {"sync", "batch"}:
        logger.warning("Unknown parse mode '%s'; defaulting to sync", mode)
        mode = "sync"
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    config = load_config()
    parquet_dir = PROJECT_ROOT / config["paths"]["staged_parquet"]
    marts_dir = PROJECT_ROOT / config["paths"]["marts"]
    marts_dir.mkdir(parents=True, exist_ok=True)

    cache_path = marts_dir / "comment_features.parquet"

    # Load runners with comments
    runners = pd.read_parquet(parquet_dir / "runners.parquet")
    has_comment: pd.DataFrame = runners[
        runners["comment"].notna() & (runners["comment"].astype(str).str.strip() != "")
    ].copy()  # type: ignore[assignment]
    has_comment = _normalise_key_types(has_comment)
    logger.info("Runners with comments: %d", len(has_comment))

    # Load existing cache
    if cache_path.exists():
        cached = _normalise_key_types(pd.read_parquet(cache_path))
        logger.info("Cached comment features: %d", len(cached))
    else:
        cached = pd.DataFrame()

    cached_keys = set(zip(cached.get("race_id", pd.Series(dtype=str)), cached.get("horse_id", pd.Series(dtype=int))))

    # Filter to uncached
    if cached_keys:
        cached_index = pd.MultiIndex.from_tuples(list(cached_keys), names=["race_id", "horse_id"])
        has_comment_index = pd.MultiIndex.from_frame(has_comment[["race_id", "horse_id"]])
        to_process: pd.DataFrame = has_comment[~has_comment_index.isin(cached_index)].copy()  # type: ignore[assignment]
    else:
        to_process = has_comment.copy()
    logger.info("Comments to process: %d", len(to_process))

    if to_process.empty:
        logger.info("All comments already cached")
        _compute_derived_features(cache_path, parquet_dir / "runners.parquet", marts_dir)
        return

    client = anthropic.Anthropic()
    if mode == "batch":
        cached = _parse_comments_batch(
            client=client,
            to_process=to_process,
            cached=cached,
            cache_path=cache_path,
            batch_size=batch_size,
            poll_interval_secs=poll_interval_secs,
            max_wait_minutes=max_wait_minutes,
        )
    else:
        cached = _parse_comments_sync(
            client=client,
            to_process=to_process,
            cached=cached,
            cache_path=cache_path,
            batch_size=batch_size,
        )

    logger.info("Comment parsing complete. Total cached: %d", len(cached))

    # Compute aggregated features over prior runs.
    _compute_derived_features(cache_path, parquet_dir / "runners.parquet", marts_dir)


def _parse_comments_sync(
    client: anthropic_mod.Anthropic,
    to_process: pd.DataFrame,
    cached: pd.DataFrame,
    cache_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    """Synchronous per-comment extraction."""
    pending: list[dict[str, object]] = []
    total = len(to_process)

    for i, row in enumerate(to_process.itertuples(index=False), start=1):
        extracted = _extract_comment_sync(client, str(row.comment))
        if extracted is not None:
            extracted["race_id"] = str(row.race_id)
            extracted["horse_id"] = int(row.horse_id)
            pending.append(extracted)

        if len(pending) >= batch_size:
            cached = _append_cache_entries(cached, pending, cache_path)
            pending = []
            logger.info("Processed %d / %d comments (sync)", i, total)

    cached = _append_cache_entries(cached, pending, cache_path)
    return cached


def _parse_comments_batch(
    client: anthropic_mod.Anthropic,
    to_process: pd.DataFrame,
    cached: pd.DataFrame,
    cache_path: Path,
    batch_size: int,
    poll_interval_secs: int,
    max_wait_minutes: int,
) -> pd.DataFrame:
    """Asynchronous extraction via Anthropic Message Batches API."""
    total = len(to_process)
    submitted = 0

    for chunk_start in range(0, total, batch_size):
        chunk = to_process.iloc[chunk_start:chunk_start + batch_size].copy()
        requests: list[dict[str, object]] = []
        for row in chunk.itertuples(index=False):
            custom_id = _make_custom_id(str(row.race_id), int(row.horse_id))
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL_NAME,
                        "max_tokens": 200,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": str(row.comment)}],
                    },
                }
            )

        batch = client.messages.batches.create(requests=requests)
        logger.info(
            "Submitted batch %s: %d requests (%d/%d submitted)",
            batch.id,
            len(requests),
            min(submitted + len(requests), total),
            total,
        )

        _wait_for_batch_completion(
            client=client,
            batch_id=batch.id,
            poll_interval_secs=poll_interval_secs,
            max_wait_minutes=max_wait_minutes,
        )

        successes: list[dict[str, object]] = []
        failures = 0
        for result in client.messages.batches.results(batch.id):
            race_id, horse_id = _parse_custom_id(result.custom_id)
            if race_id is None or horse_id is None:
                failures += 1
                continue

            if getattr(result.result, "type", None) != "succeeded":
                failures += 1
                continue

            text = _extract_text_from_message(result.result.message)
            parsed = _parse_json_payload(text)
            if parsed is None:
                failures += 1
                continue

            parsed["race_id"] = race_id
            parsed["horse_id"] = horse_id
            successes.append(parsed)

        cached = _append_cache_entries(cached, successes, cache_path)
        submitted += len(requests)
        logger.info(
            "Completed batch %s: %d saved, %d failed (%d/%d total)",
            batch.id,
            len(successes),
            failures,
            submitted,
            total,
        )

    return cached


def _wait_for_batch_completion(
    client: anthropic_mod.Anthropic,
    batch_id: str,
    poll_interval_secs: int,
    max_wait_minutes: int,
) -> None:
    """Poll batch status until processing ends."""
    deadline = time.monotonic() + max_wait_minutes * 60
    poll_count = 0

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        poll_count += 1
        if poll_count == 1 or poll_count % 8 == 0:
            counts = batch.request_counts
            logger.info(
                "Batch %s status=%s (processing=%d succeeded=%d errored=%d canceled=%d expired=%d)",
                batch_id,
                batch.processing_status,
                counts.processing,
                counts.succeeded,
                counts.errored,
                counts.canceled,
                counts.expired,
            )
        if batch.processing_status == "ended":
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for batch {batch_id}")
        time.sleep(poll_interval_secs)


def _extract_comment_sync(
    client: anthropic_mod.Anthropic,
    comment: str,
) -> dict[str, object] | None:
    """Extract structured features from one comment via synchronous Messages API."""
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": comment}],
        )
        text = _extract_text_from_message(response)
        return _parse_json_payload(text)
    except Exception as e:
        logger.warning("Failed to parse comment: %s", e)
        return None


def _extract_text_from_message(message: object) -> str:
    """Extract text blocks from Anthropic message objects."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            continue

        block_type = getattr(block, "type", None)
        block_text = getattr(block, "text", None)
        if block_type == "text" and isinstance(block_text, str):
            parts.append(block_text)

    return "\n".join(parts).strip()


def _parse_json_payload(text: str) -> dict[str, object] | None:
    """Parse JSON payload from model output, stripping markdown fences when present."""
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
            stripped = stripped.rsplit("```", 1)[0].strip()
        return json.loads(stripped)
    except Exception:
        logger.warning("Invalid JSON payload from comment parser")
        return None


def _append_cache_entries(
    cached: pd.DataFrame,
    entries: list[dict[str, object]],
    cache_path: Path,
) -> pd.DataFrame:
    """Append parsed entries into cache and persist with key deduplication."""
    if not entries:
        return cached

    new_df = pd.DataFrame(entries)
    combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    combined = _normalise_key_types(combined)
    combined = combined.drop_duplicates(subset=["race_id", "horse_id"], keep="last")
    combined.to_parquet(cache_path, index=False)
    return combined


def _normalise_key_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype("string").str.strip()
    out["horse_id"] = pd.to_numeric(out["horse_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["race_id", "horse_id"])
    out = out[out["race_id"] != ""]
    out["race_id"] = out["race_id"].astype(str)
    out["horse_id"] = out["horse_id"].astype(int)
    return out


def _make_custom_id(race_id: str, horse_id: int) -> str:
    return f"{race_id}__{horse_id}"


def _parse_custom_id(custom_id: str) -> tuple[str | None, int | None]:
    if "__" not in custom_id:
        logger.warning("Unexpected custom_id format: %s", custom_id)
        return None, None
    race_id, horse_raw = custom_id.split("__", 1)
    try:
        return race_id, int(horse_raw)
    except ValueError:
        logger.warning("Invalid horse_id in custom_id: %s", custom_id)
        return None, None


def _compute_derived_features(
    cache_path: Path,
    runners_path: Path,
    marts_dir: Path,
) -> None:
    """Compute aggregated comment features: dominant_style, pct_trouble, pct_jumping_issues."""
    import duckdb

    if not cache_path.exists():
        logger.warning("Comment cache missing; skipping derived comment features")
        return

    comments = pd.read_parquet(cache_path)
    if comments.empty:
        logger.warning("Comment cache is empty; skipping derived comment features")
        return

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
            (SELECT AVG(CASE WHEN COALESCE(CAST(h.trouble_in_running AS BOOLEAN), FALSE) THEN 1.0 ELSE 0 END)
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
