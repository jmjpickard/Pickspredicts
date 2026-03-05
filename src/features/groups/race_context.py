"""G5: Race context features — field size, class, type, distance/going encoding."""

import duckdb
import pandas as pd


def race_context(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute race-level context features from the enriched view."""
    return con.sql("""
        SELECT
            race_id,
            horse_id,
            field_size,
            CAST(is_handicap AS INTEGER) AS is_handicap,
            CASE race_type
                WHEN 'chase' THEN 0
                WHEN 'hurdle' THEN 1
                WHEN 'nh_flat' THEN 2
                ELSE NULL
            END AS race_type_encoded,
            CAST(
                REGEXP_EXTRACT(race_class, '([0-9]+)', 1) AS INTEGER
            ) AS race_class_num,
            CASE
                WHEN pattern ILIKE '%grade 1%' THEN 1 ELSE 0
            END AS is_grade1,
            CASE
                WHEN pattern ILIKE '%grade 2%' THEN 1 ELSE 0
            END AS is_grade2,
            CASE
                WHEN pattern ILIKE '%grade 3%' THEN 1 ELSE 0
            END AS is_grade3,
            distance_band,
            going_bucket
        FROM enriched
    """).df()
