"""G3: Connection features — trainer/jockey win rates, festival records, combo stats."""

import duckdb
import pandas as pd

_MIN_SAMPLE = 3


def connections(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute trainer/jockey connection features from the enriched view."""
    return con.sql(f"""
        SELECT
            t.race_id,
            t.horse_id,
            -- trainer win% over rolling calendar windows
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 14 DAY
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_winpct_14d,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 30 DAY
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_winpct_30d,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 90 DAY
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_winpct_90d,
            -- jockey win% over rolling calendar windows
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 14 DAY
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS jockey_winpct_14d,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 30 DAY
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS jockey_winpct_30d,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 90 DAY
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS jockey_winpct_90d,
            -- trainer festival record (Cheltenham in March)
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.race_date < t.race_date
               AND h.course = 'Cheltenham'
               AND EXTRACT(MONTH FROM h.race_date) = 3
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS trainer_festival_winpct,
            -- jockey festival record
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
               AND h.course = 'Cheltenham'
               AND EXTRACT(MONTH FROM h.race_date) = 3
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS jockey_festival_winpct,
            -- trainer+jockey combo career win%
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.trainer_id = t.trainer_id
               AND h.jockey_id = t.jockey_id
               AND h.race_date < t.race_date
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS combo_winpct
        FROM enriched t
    """).df()
