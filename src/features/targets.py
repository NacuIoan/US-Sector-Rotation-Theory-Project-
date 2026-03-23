"""
targets.py — Build prediction target variables.

Target definitions (from Phase 1 design):
- outperform_1m:   binary, 1 if sector log return > SPY log return at t+1
- outperform_3m:   binary, 1 if cumulative sector return > SPY over t+1 to t+3
- excess_return_1m: continuous, sector log return - SPY log return at t+1

CRITICAL: targets use FUTURE returns (shift -1, -3) aligned with features at time t.
This is intentional — they represent what we are trying to predict.
"""

import numpy as np
import pandas as pd

from src.utils.config import SECTOR_ETFS, BENCHMARK


def build_targets(
    log_returns: pd.DataFrame,
    sectors: list[str] | None = None,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """
    Build forward-looking prediction targets for each sector.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log returns with columns for sectors + benchmark.
    sectors : list[str] or None
        Sector tickers to build targets for. None = all SECTOR_ETFS.
    benchmark : str
        Benchmark ticker. Default 'SPY'.

    Returns
    -------
    pd.DataFrame
        Target columns indexed by the CURRENT month (features month).
        The last 1–3 rows will have NaN where forward data is unavailable.
    """
    if sectors is None:
        sectors = [s for s in SECTOR_ETFS if s in log_returns.columns]

    if benchmark not in log_returns.columns:
        raise KeyError(f"Benchmark '{benchmark}' not found in log_returns columns.")

    targets = pd.DataFrame(index=log_returns.index)

    spy_ret = log_returns[benchmark]

    for ticker in sectors:
        sector_ret = log_returns[ticker]

        # --- 1-month ahead excess return (continuous) ---
        excess_1m = sector_ret.shift(-1) - spy_ret.shift(-1)
        targets[f"{ticker}_excess_1m"] = excess_1m

        # --- 1-month ahead outperformance (binary) ---
        targets[f"{ticker}_outperform_1m"] = (excess_1m > 0).astype(int)

        # --- 3-month ahead cumulative outperformance (binary) ---
        # Cumulative return over t+1, t+2, t+3
        cum_sector = sector_ret.shift(-1) + sector_ret.shift(-2) + sector_ret.shift(-3)
        cum_spy    = spy_ret.shift(-1)    + spy_ret.shift(-2)    + spy_ret.shift(-3)
        targets[f"{ticker}_outperform_3m"] = ((cum_sector - cum_spy) > 0).astype(int)

    # Mark rows with NaN where forward data doesn't exist
    # (the last 1–3 rows naturally become NaN because of the shifts)
    print(f"[TARGETS] Built {targets.shape[1]} target columns "
          f"for {len(sectors)} sectors.")
    return targets
