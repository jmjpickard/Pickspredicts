"""G9: Ratings vs field — within-race relative strength + RPR trend."""

import duckdb
import pandas as pd


def ratings_vs_field(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute within-race relative rating features from the enriched view."""
    return con.sql("""
        WITH horse_ratings AS (
            SELECT
                race_id,
                horse_id,
                official_rating,
                MAX(rpr) OVER (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) AS rpr_best_last3,
                LAG(topspeed) OVER (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                ) AS ts_current,
                REGR_SLOPE(rpr, run_seq) OVER (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) AS rpr_trend_last5
            FROM enriched
        )
        SELECT
            race_id,
            horse_id,
            official_rating
                - AVG(official_rating) OVER (PARTITION BY race_id)
                AS or_vs_field,
            rpr_best_last3
                - AVG(rpr_best_last3) OVER (PARTITION BY race_id)
                AS rpr_vs_field,
            ts_current
                - AVG(ts_current) OVER (PARTITION BY race_id)
                AS ts_vs_field,
            rpr_trend_last5
        FROM horse_ratings
    """).df()
