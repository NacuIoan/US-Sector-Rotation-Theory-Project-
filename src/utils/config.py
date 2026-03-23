"""
config.py — Central configuration for the Sector Rotation project.

All tickers, FRED series IDs, date ranges, and path constants
are defined here so that every other module imports from one place.
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/utils/config.py) ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"

# ── ETF tickers ───────────────────────────────────────────────────────
SECTOR_ETFS = [
    "XLB",   # Materials
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer Staples
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLC",   # Communication Services
]

BENCHMARK = "SPY"

ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

# Human-readable sector names (same order as SECTOR_ETFS)
SECTOR_NAMES = {
    "XLB":  "Materials",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLK":  "Technology",
    "XLP":  "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLC":  "Communication Services",
    "SPY":  "S&P 500 (Benchmark)",
}

# ── FRED macro series ────────────────────────────────────────────────
# Mapping: descriptive column name → FRED series ID
FRED_SERIES = {
    "CPI":                  "CPIAUCSL",    # CPI All Urban Consumers
    "Core_CPI":             "CPILFESL",    # Core CPI (ex food & energy)
    "Fed_Funds_Rate":       "FEDFUNDS",    # Effective Federal Funds Rate
    "Treasury_10Y":         "GS10",        # 10-Year Treasury Yield
    "Treasury_2Y":          "GS2",         # 2-Year Treasury Yield
    "Unemployment":         "UNRATE",      # Unemployment Rate
    "Industrial_Production":"INDPRO",      # Industrial Production Index
    "Retail_Sales":         "RSXFS",       # Advance Retail Sales
    "Consumer_Sentiment":   "UMCSENT",     # U. of Michigan Consumer Sentiment
    "NBER_Recession":       "USREC",       # NBER Recession Indicator
    "Housing_Starts":       "HOUST",       # Housing Starts
}

# ── Date range ────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE   = None          # None → download up to the latest available date

# ── Reproducibility ──────────────────────────────────────────────────
RANDOM_STATE = 42

# ── File names for raw data ──────────────────────────────────────────
ETF_RAW_FILE   = RAW_DIR / "etf_prices.parquet"
MACRO_RAW_FILE = RAW_DIR / "macro_data.parquet"
