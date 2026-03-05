"""G4: Pedigree features — sire performance by course, going, and distance."""

import duckdb
import pandas as pd

_MIN_SAMPLE = 5


def pedigree(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute sire-based pedigree features from the enriched view.

    Uses full history (no rolling window) but excludes current row date
    for leakage prevention.
    """
    return con.sql(f"""
        SELECT
            t.race_id,
            t.horse_id,
            -- sire win% at Cheltenham (all-time, pre-race)
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.sire_id = t.sire_id
               AND h.race_date < t.race_date
               AND h.course = 'Cheltenham'
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS sire_cheltenham_winpct,
            -- sire win% on same going bucket
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.sire_id = t.sire_id
               AND h.race_date < t.race_date
               AND h.going_bucket = t.going_bucket
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS sire_going_winpct,
            -- sire win% at same distance band
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.sire_id = t.sire_id
               AND h.race_date < t.race_date
               AND h.distance_band = t.distance_band
             HAVING COUNT(*) >= {_MIN_SAMPLE}
            ) AS sire_dist_winpct
        FROM enriched t
    """).df()
