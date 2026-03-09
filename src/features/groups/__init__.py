"""Shared constants and SQL snippets for feature groups."""

# Going bucket encoding (0=good, 1=good_to_soft, 2=soft, 3=heavy).
# Missing/blank going stays NULL to avoid silently treating unknown as "good".
GOING_BUCKET_SQL = """
CASE
    WHEN {going} IS NULL OR TRIM(CAST({going} AS VARCHAR)) = '' THEN NULL
    WHEN LOWER({going}) LIKE '%heavy%' THEN 3
    WHEN LOWER({going}) LIKE '%soft%' AND LOWER({going}) NOT LIKE '%good to soft%' THEN 2
    WHEN LOWER({going}) LIKE '%soft%' OR LOWER({going}) LIKE '%yielding%' THEN 1
    ELSE 0
END
""".strip()

# Distance band encoding (0=short, 1=medium, 2=long, 3=marathon)
DISTANCE_BAND_SQL = """
CASE
    WHEN {dist} < 3200 THEN 0
    WHEN {dist} < 4000 THEN 1
    WHEN {dist} < 4800 THEN 2
    ELSE 3
END
""".strip()

# The enriched view created by the orchestrator joins runners + races
# and adds: race_date, race_type, field_size, is_handicap, going, pattern,
# race_class, distance_meters, track_direction, going_bucket, distance_band, run_seq
ENRICHED_VIEW_SQL = f"""
CREATE OR REPLACE VIEW enriched AS
SELECT
    run.*,
    CAST(run.date AS DATE) AS race_date,
    r.race_type,
    r.field_size,
    r.is_handicap,
    r.going,
    r.pattern,
    r."class" AS race_class,
    COALESCE(r.distance_m, CAST(r.distance_f * 201.168 AS INTEGER)) AS distance_meters,
    r.track_direction,
    {GOING_BUCKET_SQL.format(going='r.going')} AS going_bucket,
    {DISTANCE_BAND_SQL.format(dist='COALESCE(r.distance_m, CAST(r.distance_f * 201.168 AS INTEGER))')} AS distance_band,
    ROW_NUMBER() OVER (
        PARTITION BY run.horse_id ORDER BY run.date, run.race_id
    ) AS run_seq
FROM runners run
JOIN races r ON run.race_id = r.race_id
"""
