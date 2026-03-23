"""
returns.py — Compute simple, log, and excess returns from price data.

Covers Course Requirement 4 (statistical processing, aggregation).

Formulas
--------
- Simple return:  r_t = (P_t / P_{t-1}) - 1
- Log return:     r_t = ln(P_t / P_{t-1})
- Excess return:  sector return - SPY return  (for each period)
"""

import numpy as np
import pandas as pd


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple (arithmetic) monthly returns.

    Parameters
    ----------
    prices : pd.DataFrame
        End-of-month adjusted-close prices, one column per ticker.

    Returns
    -------
    pd.DataFrame
        Simple returns with the same columns. First row is NaN (dropped).
    """
    returns = prices.pct_change()
    returns.dropna(how="all", inplace=True)
    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log (continuously compounded) monthly returns.

    Parameters
    ----------
    prices : pd.DataFrame
        End-of-month adjusted-close prices.

    Returns
    -------
    pd.DataFrame
        Log returns. First row is NaN (dropped).
    """
    log_ret = np.log(prices / prices.shift(1))
    log_ret.dropna(how="all", inplace=True)
    return log_ret


def compute_excess_returns(
    returns: pd.DataFrame,
    benchmark_col: str = "SPY",
) -> pd.DataFrame:
    """
    Compute excess returns: each sector minus the benchmark.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series that includes a benchmark column.
    benchmark_col : str
        Name of the benchmark column (default 'SPY').

    Returns
    -------
    pd.DataFrame
        Excess returns for every column EXCEPT the benchmark.
    """
    if benchmark_col not in returns.columns:
        raise KeyError(f"Benchmark column '{benchmark_col}' not found in DataFrame.")

    sector_cols = [c for c in returns.columns if c != benchmark_col]
    excess = returns[sector_cols].subtract(returns[benchmark_col], axis=0)
    return excess
