"""G10: Enhanced features — sex encoding, festival experience, class context."""

import duckdb
import pandas as pd


def enhanced(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute sex, festival experience, and class context features."""
    return con.sql("""
        SELECT
            t.race_id,
            t.horse_id,

            -- Sex encoding: G=0, M/F=1 (female), C/H=2 (intact male)
            CASE
                WHEN t.sex IN ('M', 'F') THEN 1
                WHEN t.sex IN ('C', 'H') THEN 2
                ELSE 0
            END AS sex_encoded,

            -- Festival experience: prior March/April starts at Cheltenham or Aintree
            (SELECT COUNT(*)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.course IN ('Cheltenham', 'Aintree')
               AND EXTRACT(MONTH FROM h.race_date) IN (3, 4)
            ) AS festival_starts,

            -- Class drop from Grade 1: running below top level after G1 experience
            CASE WHEN EXISTS (
                SELECT 1 FROM enriched h
                WHERE h.horse_id = t.horse_id
                  AND h.race_date < t.race_date
                  AND h.pattern ILIKE '%grade 1%'
            ) AND NOT (t.pattern ILIKE '%grade 1%')
            THEN 1 ELSE 0
            END AS class_drop_from_grade1,

            -- Days since last win (improved: capped at 999, NULL → 999 for model)
            COALESCE(
                (SELECT t.race_date - MAX(h.race_date)
                 FROM enriched h
                 WHERE h.horse_id = t.horse_id
                   AND h.race_date < t.race_date
                   AND h.finish_position = 1
                ),
                999
            ) AS days_since_last_win_capped

        FROM enriched t
    """).df()
