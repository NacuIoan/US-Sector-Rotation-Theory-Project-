"""
macro_features.py — Derive macro indicators and merge with ETF data.

Covers Course Requirement 5 (merge / join operations).

Derived features:
- CPI YoY %:  12-month percentage change of CPI
- Core CPI YoY %: same for Core CPI
- IP MoM %:  1-month percentage change of Industrial Production
- Yield Spread:  10-Year Treasury – 2-Year Treasury
- IP MoM 3MA:  3-month moving average of IP MoM (smoothed growth signal)

All features are lagged by at least 1 month to avoid look-ahead bias.
"""

import pandas as pd


# ── Individual derived series ────────────────────────────────────────

def compute_cpi_yoy(macro_df: pd.DataFrame) -> pd.Series:
    """CPI year-over-year percentage change."""
    return macro_df["CPI"].pct_change(periods=12) * 100


def compute_core_cpi_yoy(macro_df: pd.DataFrame) -> pd.Series:
    """Core CPI year-over-year percentage change."""
    return macro_df["Core_CPI"].pct_change(periods=12) * 100


def compute_ip_mom(macro_df: pd.DataFrame) -> pd.Series:
    """Industrial Production month-over-month percentage change."""
    return macro_df["Industrial_Production"].pct_change(periods=1) * 100


def compute_yield_spread(macro_df: pd.DataFrame) -> pd.Series:
    """10-Year minus 2-Year Treasury yield spread."""
    return macro_df["Treasury_10Y"] - macro_df["Treasury_2Y"]


# ── Orchestrator: build all macro features ───────────────────────────

def build_macro_features(
    macro_df: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a full set of macro features from raw FRED data.

    Steps:
    1. Compute derived indicators (YoY, MoM, spread).
    2. Create a 3-month moving average of IP MoM (growth signal).
    3. Lag all features by 1 and 2 months for look-ahead-safe modeling.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Raw macro data with columns matching config.FRED_SERIES keys.
    lags : list[int] or None
        Lag periods to create. Default [1, 2].

    Returns
    -------
    pd.DataFrame
        Macro features including lagged columns. Index = Date.
    """
    if lags is None:
        lags = [1, 2]

    feat = pd.DataFrame(index=macro_df.index)

    # --- Raw levels (keep a subset useful for modeling) ---
    feat["Fed_Funds_Rate"]      = macro_df["Fed_Funds_Rate"]
    feat["Treasury_10Y"]        = macro_df["Treasury_10Y"]
    feat["Treasury_2Y"]         = macro_df["Treasury_2Y"]
    feat["Unemployment"]        = macro_df["Unemployment"]
    feat["Consumer_Sentiment"]  = macro_df["Consumer_Sentiment"]
    feat["NBER_Recession"]      = macro_df["NBER_Recession"]
    feat["Housing_Starts"]      = macro_df["Housing_Starts"]
    feat["Retail_Sales"]        = macro_df["Retail_Sales"]

    # --- Derived indicators ---
    feat["CPI_YoY"]       = compute_cpi_yoy(macro_df)
    feat["Core_CPI_YoY"]  = compute_core_cpi_yoy(macro_df)
    feat["IP_MoM"]         = compute_ip_mom(macro_df)
    feat["Yield_Spread"]   = compute_yield_spread(macro_df)

    # --- Smoothed growth signal (3-month MA of IP MoM) ---
    feat["IP_MoM_3MA"] = feat["IP_MoM"].rolling(window=3).mean()

    # --- Lagged features (for look-ahead-safe modeling) ---
    base_cols = [c for c in feat.columns if c != "NBER_Recession"]
    for lag in lags:
        for col in base_cols:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    return feat


# ── Merge ETF returns + macro features ───────────────────────────────

def merge_etf_macro(
    etf_returns: pd.DataFrame,
    macro_features: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge ETF returns with macro features on their Date index.

    This is the primary merge/join operation (Course Req. 5).

    Parameters
    ----------
    etf_returns : pd.DataFrame
        Monthly returns (simple or log) with DatetimeIndex.
    macro_features : pd.DataFrame
        Macro features with DatetimeIndex (from build_macro_features).
    how : str
        Join type. Default 'inner' to keep only complete rows.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ETF return columns + macro feature columns.
    """
    merged = etf_returns.join(macro_features, how=how)
    print(f"[MERGE] ETF ({etf_returns.shape}) + Macro ({macro_features.shape}) "
          f"→ merged ({merged.shape})")
    return merged
