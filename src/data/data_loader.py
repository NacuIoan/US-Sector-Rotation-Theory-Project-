"""
data_loader.py — Unified data loading interface.

Reads ETF and macro data from Parquet on disk.
If the files don't exist yet (or force_download=True),
triggers the download functions and saves the results.

Also provides `load_processed_dataset()` which orchestrates the full
Phase 3 pipeline: raw data → returns → macro features → merge → targets.

Usage:
    from src.data.data_loader import load_etf_data, load_macro_data, load_all_data
    from src.data.data_loader import load_processed_dataset

    etf_df, macro_df = load_all_data()
    full_df = load_processed_dataset()
"""

import pandas as pd

from src.utils.config import ETF_RAW_FILE, MACRO_RAW_FILE, PROCESSED_DIR


def load_etf_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load ETF monthly prices from disk, downloading first if necessary.

    Parameters
    ----------
    force_download : bool
        If True, re-download even when the file exists on disk.

    Returns
    -------
    pd.DataFrame
        Month-end adjusted-close prices with one column per ticker.
    """
    if not force_download and ETF_RAW_FILE.exists():
        print(f"[LOADER] Reading ETF data from {ETF_RAW_FILE}")
        return pd.read_parquet(ETF_RAW_FILE)

    # File missing or forced refresh → download
    print("[LOADER] ETF file not found — downloading …")
    from src.data.download_etf import download_etf_data, save_etf_data

    df = download_etf_data()
    save_etf_data(df)
    return df


def load_macro_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load macro data from disk, downloading first if necessary.

    Parameters
    ----------
    force_download : bool
        If True, re-download even when the file exists on disk.

    Returns
    -------
    pd.DataFrame
        Month-end macro indicators with one column per series.
    """
    if not force_download and MACRO_RAW_FILE.exists():
        print(f"[LOADER] Reading macro data from {MACRO_RAW_FILE}")
        return pd.read_parquet(MACRO_RAW_FILE)

    print("[LOADER] Macro file not found — downloading …")
    from src.data.download_macro import download_macro_data, save_macro_data

    df = download_macro_data()
    save_macro_data(df)
    return df


def load_all_data(
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: load both ETF and macro DataFrames.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (etf_df, macro_df)
    """
    etf_df   = load_etf_data(force_download=force_download)
    macro_df = load_macro_data(force_download=force_download)
    return etf_df, macro_df


# ── Full processed dataset (Phase 3 pipeline) ───────────────────────

PROCESSED_FILE = PROCESSED_DIR / "sector_data.parquet"


def load_processed_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the fully processed sector dataset, building it from raw data
    if it doesn't exist or if force_rebuild=True.

    Pipeline steps:
    1. Load raw ETF prices + raw macro data
    2. Compute simple & log returns
    3. Build macro features (derived indicators, lags)
    4. Merge ETF returns with macro features (inner join on Date)
    5. Build forward-looking prediction targets
    6. Handle missing values (forward fill, then drop remaining)
    7. Save to data/processed/sector_data.parquet

    Parameters
    ----------
    force_rebuild : bool
        If True, re-run the pipeline even if the processed file exists.

    Returns
    -------
    pd.DataFrame
        Full processed dataset ready for EDA, modeling, and dashboard.
    """
    if not force_rebuild and PROCESSED_FILE.exists():
        print(f"[LOADER] Reading processed data from {PROCESSED_FILE}")
        return pd.read_parquet(PROCESSED_FILE)

    print("[LOADER] Building processed dataset from raw data …")

    # --- Lazy imports (avoid circular imports / unnecessary deps) ---
    from src.features.returns import (
        compute_simple_returns,
        compute_log_returns,
        compute_excess_returns,
    )
    from src.features.macro_features import build_macro_features, merge_etf_macro
    from src.features.targets import build_targets
    from src.utils.preprocessing import handle_missing, report_missing
    from src.utils.io_helpers import save_processed

    # 1. Load raw data
    etf_prices = load_etf_data()
    macro_raw  = load_macro_data()

    # 2. Compute returns
    simple_ret = compute_simple_returns(etf_prices)
    log_ret    = compute_log_returns(etf_prices)
    excess_ret = compute_excess_returns(log_ret)

    # Rename columns to avoid clashes when merging
    simple_ret.columns = [f"{c}_simple_ret" for c in simple_ret.columns]
    log_ret_named      = log_ret.rename(columns={c: f"{c}_log_ret" for c in log_ret.columns})
    excess_ret.columns = [f"{c}_excess_ret" for c in excess_ret.columns]

    # Combine all return series
    all_returns = simple_ret.join(log_ret_named, how="outer").join(excess_ret, how="outer")

    # 3. Build macro features (with lags)
    macro_feat = build_macro_features(macro_raw)

    # 4. Merge returns + macro features
    merged = merge_etf_macro(all_returns, macro_feat, how="inner")

    # 5. Build targets (using original log returns, not renamed)
    targets = build_targets(log_ret)
    merged = merged.join(targets, how="left")

    # 6. Handle missing values
    #    - Macro-derived columns may have NaN from YoY/lag calculations:
    #      forward-fill then drop rows where macro lags are still NaN
    #      (first ~13 rows lost to 12-month YoY + 2-month lag).
    #    - ETF columns for XLC (inception 2018) and XLRE (inception 2015)
    #      have structural NaN before their inception — these are NOT data
    #      errors, so we leave them as NaN and do NOT drop rows for them.
    missing = report_missing(merged)
    if not missing.empty:
        print("[LOADER] Missing values before cleaning:")
        print(missing.head(15))

    # Forward-fill only macro feature columns (not ETF returns/targets)
    macro_cols = [c for c in macro_feat.columns if c in merged.columns]
    merged[macro_cols] = merged[macro_cols].ffill()

    # Drop rows where critical macro lag columns are still NaN
    # (these are the leading rows lost to YoY and lag calculations)
    lag_cols = [c for c in merged.columns if c.endswith("_lag2")]
    if lag_cols:
        merged = merged.dropna(subset=lag_cols)

    print(f"[LOADER] After cleaning: {merged.isna().sum().sum()} NaN remaining "
          f"(pre-inception ETFs only)")

    # 7. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_processed(merged, "sector_data")

    print(f"[LOADER] Processed dataset: {merged.shape[0]} rows × {merged.shape[1]} cols")
    return merged

