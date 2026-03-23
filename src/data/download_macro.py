"""
download_macro.py — Download macroeconomic indicators from FRED.

Uses the fredapi library with an API key stored in .env.
Each FRED series is fetched individually, then all series are combined
into a single DataFrame and saved as Parquet.

Run standalone:
    python -m src.data.download_macro
"""

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from src.utils.config import (
    FRED_SERIES,
    START_DATE,
    END_DATE,
    MACRO_RAW_FILE,
    PROJECT_ROOT,
)


def _get_fred_client() -> Fred:
    """
    Load FRED API key from .env and return a Fred client.

    Raises
    ------
    ValueError
        If FRED_API_KEY is not set or is still the placeholder value.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FRED_API_KEY", "")

    if not api_key or api_key == "your_fred_api_key_here":
        raise ValueError(
            "FRED_API_KEY is not configured.  "
            "Edit the .env file in the project root and paste your key.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=api_key)


def download_macro_data(
    series_dict: dict[str, str] = FRED_SERIES,
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    """
    Download multiple FRED series and combine into one DataFrame.

    Parameters
    ----------
    series_dict : dict[str, str]
        Mapping of descriptive column name → FRED series ID.
    start : str
        Start date in YYYY-MM-DD.
    end : str or None
        End date (None = latest available).

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with one column per series.
        Index is end-of-month DatetimeIndex named "Date".
    """
    fred = _get_fred_client()
    collected: dict[str, pd.Series] = {}

    for col_name, series_id in series_dict.items():
        try:
            print(f"[MACRO] Fetching {series_id} ({col_name}) …")
            s = fred.get_series(series_id, observation_start=start,
                                observation_end=end)
            collected[col_name] = s
        except Exception as exc:
            print(f"[MACRO] ⚠  Failed to fetch {series_id}: {exc}")
            continue

    if not collected:
        raise RuntimeError("No FRED series were downloaded successfully.")

    # Combine into a single DataFrame
    df = pd.DataFrame(collected)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Most FRED series are already monthly; resample to ensure alignment
    df = df.resample("ME").last()

    print(f"[MACRO] Got {len(df)} monthly observations, "
          f"{df.shape[1]} series.")
    return df


def save_macro_data(df: pd.DataFrame, path=MACRO_RAW_FILE) -> None:
    """Save macro DataFrame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    print(f"[MACRO] Saved → {path}")


# ── CLI entry point ──────────────────────────────────────────────────
def main() -> None:
    """Download all FRED macro data and save to disk."""
    df = download_macro_data()
    save_macro_data(df)
    print("[MACRO] Done.")


if __name__ == "__main__":
    main()
