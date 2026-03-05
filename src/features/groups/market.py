"""G7: Market features — Betfair exchange pre-race pricing signals.

All features are pre-race (morning/pre-play), no in-play leakage.
"""

import duckdb
import numpy as np
import pandas as pd


def market(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute market features from betfair_historical table."""
    df: pd.DataFrame = con.sql("""
        SELECT
            b.race_id,
            b.horse_id,
            b.morning_wap,
            b.pre_min,
            b.pre_vol
        FROM betfair_historical b
    """).df()

    if df.empty:
        return pd.DataFrame(columns=["race_id", "horse_id"])

    # market_implied_prob: 1 / morning_wap
    df["market_implied_prob"] = np.where(
        df["morning_wap"].notna() & (df["morning_wap"] > 0),
        1.0 / df["morning_wap"],
        np.nan,
    )

    # market_rank: rank by morning_wap within race (1 = favourite = lowest WAP)
    df["market_rank"] = df.groupby("race_id")["morning_wap"].rank(method="min")

    # pre_price_move: morning_wap / pre_min (drift/shortening signal)
    df["pre_price_move"] = np.where(
        df["morning_wap"].notna() & df["pre_min"].notna() & (df["pre_min"] > 0),
        df["morning_wap"] / df["pre_min"],
        np.nan,
    )

    # market_confidence: log(pre_vol + 1) (liquidity proxy)
    df["market_confidence"] = np.where(
        df["pre_vol"].notna(),
        np.log(df["pre_vol"] + 1),
        np.nan,
    )

    out_cols = ["race_id", "horse_id", "market_implied_prob", "market_rank",
                "pre_price_move", "market_confidence"]
    return df[out_cols].copy()  # type: ignore[return-value]
