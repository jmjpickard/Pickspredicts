"""G8: Runner profile features — SP rank, weight, course-distance, class change."""

import duckdb
import pandas as pd


def runner_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute runner profile features from the enriched view."""
    return con.sql("""
        SELECT
            t.race_id,
            t.horse_id,

            -- SP rank (1 = favourite)
            RANK() OVER (PARTITION BY t.race_id ORDER BY t.sp_decimal) AS sp_rank,

            -- Weight carried (raw lbs)
            t.weight_lbs AS weight_carried,

            -- Weight vs field average
            t.weight_lbs - AVG(t.weight_lbs)
                OVER (PARTITION BY t.race_id) AS weight_vs_field_avg,

            -- Course-distance winner (any prior win at same course + distance band)
            CASE WHEN EXISTS (
                SELECT 1 FROM enriched h
                WHERE h.horse_id = t.horse_id
                  AND h.race_date < t.race_date
                  AND h.course = t.course
                  AND h.distance_band = t.distance_band
                  AND h.finish_position = 1
            ) THEN 1 ELSE 0 END AS course_dist_winner,

            -- Days since last win
            (SELECT t.race_date - MAX(h.race_date)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.finish_position = 1
            ) AS days_since_last_win,

            -- Class change from last run (positive = dropping down, negative = stepping up)
            CAST(REGEXP_EXTRACT(t.race_class, '([0-9]+)', 1) AS INTEGER)
            - LAG(CAST(REGEXP_EXTRACT(t.race_class, '([0-9]+)', 1) AS INTEGER)) OVER (
                PARTITION BY t.horse_id ORDER BY t.race_date, t.race_id
            ) AS class_change

        FROM enriched t
    """).df()
