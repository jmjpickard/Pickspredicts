"""G12: Horse context features — first-time indicators, field quality, OR movement."""

import duckdb
import pandas as pd


def horse_context(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """First-time flags, field strength metrics, and handicap mark movement."""
    return con.sql("""
        SELECT
            t.race_id,
            t.horse_id,

            -- First time chasing (horse has only run hurdles/flat before)
            CASE WHEN t.race_type = 'chase' AND NOT EXISTS (
                SELECT 1 FROM enriched h
                WHERE h.horse_id = t.horse_id
                  AND h.race_date < t.race_date
                  AND h.race_type = 'chase'
            ) THEN 1 ELSE 0 END AS first_time_chase,

            -- First time at this course
            CASE WHEN NOT EXISTS (
                SELECT 1 FROM enriched h
                WHERE h.horse_id = t.horse_id
                  AND h.race_date < t.race_date
                  AND h.course = t.course
            ) THEN 1 ELSE 0 END AS first_time_course,

            -- First time at this distance band
            CASE WHEN NOT EXISTS (
                SELECT 1 FROM enriched h
                WHERE h.horse_id = t.horse_id
                  AND h.race_date < t.race_date
                  AND h.distance_band = t.distance_band
            ) THEN 1 ELSE 0 END AS first_time_distance,

            -- Field quality: average OR of runners in this race
            AVG(t.official_rating) OVER (PARTITION BY t.race_id) AS field_avg_or,

            -- Field quality: max OR (how strong is the top horse)
            MAX(t.official_rating) OVER (PARTITION BY t.race_id) AS field_max_or,

            -- OR change from last run (positive = raised in weights)
            t.official_rating - LAG(t.official_rating) OVER (
                PARTITION BY t.horse_id ORDER BY t.race_date, t.race_id
            ) AS or_change_last_run

        FROM enriched t
    """).df()
