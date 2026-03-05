"""G2: Horse form features — career stats, recency, going/distance/course suitability."""

import duckdb
import pandas as pd


def horse_form(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute horse form features from the enriched view.

    Split into window-based, calendar-based, and filtered-rate queries
    then merged on (race_id, horse_id).
    """
    window_df = _window_features(con)
    calendar_df = _calendar_features(con)
    filtered_df = _filtered_win_rates(con)

    result = window_df.merge(calendar_df, on=["race_id", "horse_id"], how="left")
    result = result.merge(filtered_df, on=["race_id", "horse_id"], how="left")
    return result


def _window_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Row-ordered window features: career stats, streaks, headgear."""
    return con.sql("""
        WITH raw_windows AS (
            SELECT
                race_id,
                horse_id,
                age AS age_at_race,
                race_date - LAG(race_date) OVER w_ord AS days_since_last_run,
                COUNT(*) OVER w_career AS career_runs,
                SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END)
                    OVER w_career AS career_wins,
                SUM(CASE WHEN finish_position IS NOT NULL AND finish_position <= 3
                    THEN 1 ELSE 0 END)
                    OVER w_career AS career_places,
                SUM(CASE WHEN race_type = 'chase' THEN 1 ELSE 0 END)
                    OVER w_career AS chase_starts,
                SUM(CASE WHEN race_type = 'hurdle' THEN 1 ELSE 0 END)
                    OVER w_career AS hurdle_starts,
                SUM(CASE WHEN finish_position IS NULL THEN 1 ELSE 0 END)
                    OVER w5 * 1.0
                    / NULLIF(COUNT(*) OVER w5, 0) AS dnf_rate_last5,
                AVG(ovr_btn) OVER (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) AS avg_btn_last3,
                REGR_SLOPE(ovr_btn, run_seq) OVER w5 AS btn_trend_last5,
                CASE
                    WHEN headgear IS DISTINCT FROM
                         LAG(headgear) OVER w_ord
                         AND LAG(headgear) OVER w_ord IS NOT NULL
                    THEN 1 ELSE 0
                END AS headgear_changed,
                CASE
                    WHEN headgear IS NOT NULL
                         AND RIGHT(CAST(headgear AS VARCHAR), 1) = '1'
                    THEN 1 ELSE 0
                END AS first_time_headgear
            FROM enriched
            WINDOW
                w_ord AS (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                ),
                w_career AS (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                w5 AS (
                    PARTITION BY horse_id ORDER BY race_date, race_id
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                )
        )
        SELECT
            *,
            career_wins * 1.0 / NULLIF(career_runs, 0) AS win_rate_overall,
            career_places * 1.0 / NULLIF(career_runs, 0) AS place_rate_overall
        FROM raw_windows
    """).df()


def _calendar_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Calendar-window counts: runs in last 90/365 days."""
    return con.sql("""
        SELECT
            t.race_id,
            t.horse_id,
            (SELECT COUNT(*)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 90 DAY
            ) AS runs_last_90d,
            (SELECT COUNT(*)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.race_date >= t.race_date - INTERVAL 365 DAY
            ) AS runs_last_365d
        FROM enriched t
    """).df()


def _filtered_win_rates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Win rates filtered by going bucket, distance band, and course."""
    return con.sql("""
        SELECT
            t.race_id,
            t.horse_id,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.going_bucket = t.going_bucket
             HAVING COUNT(*) >= 3
            ) AS win_rate_going_bucket,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.distance_band = t.distance_band
             HAVING COUNT(*) >= 3
            ) AS win_rate_dist_band,
            (SELECT
                SUM(CASE WHEN h.finish_position = 1 THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
             FROM enriched h
             WHERE h.horse_id = t.horse_id
               AND h.race_date < t.race_date
               AND h.course = t.course
             HAVING COUNT(*) >= 3
            ) AS win_rate_course
        FROM enriched t
    """).df()
