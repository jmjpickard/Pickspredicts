"""G1: Rating features — current OR/RPR/TS, rolling bests, trends."""

import duckdb
import pandas as pd


def ratings(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute rating features from the enriched view."""
    return con.sql("""
        SELECT
            race_id,
            horse_id,
            official_rating AS or_current,
            LAG(rpr) OVER w_all AS rpr_current,
            LAG(topspeed) OVER w_all AS ts_current,
            MAX(official_rating) OVER w3 AS or_best_last3,
            MAX(official_rating) OVER w5 AS or_best_last5,
            MAX(rpr) OVER w3 AS rpr_best_last3,
            MAX(rpr) OVER w5 AS rpr_best_last5,
            official_rating - LAG(rpr) OVER w_all AS or_rpr_diff,
            REGR_SLOPE(official_rating, run_seq) OVER w5 AS or_trend_last5
        FROM enriched
        WINDOW
            w_all AS (
                PARTITION BY horse_id ORDER BY race_date, race_id
            ),
            w3 AS (
                PARTITION BY horse_id ORDER BY race_date, race_id
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ),
            w5 AS (
                PARTITION BY horse_id ORDER BY race_date, race_id
                ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
            )
    """).df()
