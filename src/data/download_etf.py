"""
download_etf.py — Download sector ETF + benchmark prices from Yahoo Finance.

Uses yfinance to fetch daily adjusted-close prices for all sector ETFs and SPY,
then resamples to month-end frequency and saves as Parquet.

Run standalone:
    python -m src.data.download_etf
"""

import pandas as pd
import yfinance as yf

from src.utils.config import ALL_TICKERS, START_DATE, END_DATE, RAW_DIR, ETF_RAW_FILE


def download_etf_data(
    tickers: list[str] = ALL_TICKERS,
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for the given tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download.
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date (None = today).

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and one column per ticker,
        resampled to month-end frequency (last trading day of each month).
    """
    print(f"[ETF] Downloading {len(tickers)} tickers from {start} …")

    # yfinance ≥0.2.18 returns a DataFrame with multi-level columns
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # --- Extract close prices ---
    # With auto_adjust=True, "Close" is already the adjusted close.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker edge case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Ensure column order matches the tickers list
    prices = prices[[t for t in tickers if t in prices.columns]]

    # --- Resample to month-end (last observation per month) ---
    prices.index = pd.to_datetime(prices.index)
    monthly = prices.resample("ME").last()
    monthly.index.name = "Date"

    # Drop rows where ALL tickers are NaN (shouldn't happen, but safe)
    monthly.dropna(how="all", inplace=True)

    print(f"[ETF] Got {len(monthly)} monthly observations, "
          f"{monthly.shape[1]} tickers.")
    return monthly


def save_etf_data(df: pd.DataFrame, path=ETF_RAW_FILE) -> None:
    """Save ETF price DataFrame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    print(f"[ETF] Saved → {path}")


# ── CLI entry point ──────────────────────────────────────────────────
def main() -> None:
    """Download all ETF data and save to disk."""
    df = download_etf_data()
    save_etf_data(df)
    print("[ETF] Done.")


if __name__ == "__main__":
    main()
