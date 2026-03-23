"""
data_loader.py — Unified data loading interface.

Reads ETF and macro data from Parquet on disk.
If the files don't exist yet (or force_download=True),
triggers the download functions and saves the results.

Usage:
    from src.data.data_loader import load_etf_data, load_macro_data, load_all_data

    etf_df   = load_etf_data()
    macro_df = load_macro_data()
    # or
    etf_df, macro_df = load_all_data()
"""

import pandas as pd

from src.utils.config import ETF_RAW_FILE, MACRO_RAW_FILE


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
