"""G11: Extended connection features — trainer/jockey by class, course, race type."""

import duckdb
import pandas as pd

_MIN_SAMPLE = 3


def connections_extended(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Trainer/jockey win rates sliced by class level, course, and race type."""
    return con.sql(f"""
        SELECT
            t.race_id,
            t.horse_id,

            -- Trainer win% at same class level (e.g. Class 1 = Graded races)
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND CAST(REGEXP_EXTRACT(h.race_class, '([0-9]+)', 1) AS INTEGER)
                 = CAST(REGEXP_EXTRACT(t.race_class, '([0-9]+)', 1) AS INTEGER)
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_winpct_class,

            -- Trainer win% at this course (all-time)
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND h.course = t.course
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_winpct_course,

            -- Jockey win% at this course (all-time)
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
               AND h.course = t.course
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS jockey_winpct_course,

            -- Trainer win% by race type (chase / hurdle / nh_flat)
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND h.race_type = t.race_type
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_winpct_race_type,

            -- Jockey win% by race type
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
               AND h.race_type = t.race_type
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS jockey_winpct_race_type

        FROM enriched t
    """).df()
