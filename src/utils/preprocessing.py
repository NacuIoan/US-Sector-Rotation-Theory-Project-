"""
preprocessing.py — Missing-value handling, outlier detection, and scaling.

Covers Course Requirements 2 (missing & extreme values) and 3 (scaling methods).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ── Missing Values ───────────────────────────────────────────────────

def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values per column.

    Returns
    -------
    pd.DataFrame
        Columns: count, pct — sorted descending by count.
    """
    total = df.isna().sum()
    pct = (total / len(df)) * 100
    report = pd.DataFrame({"count": total, "pct": pct})
    return report[report["count"] > 0].sort_values("count", ascending=False)


def handle_missing(
    df: pd.DataFrame,
    method: str = "ffill",
    drop_remaining: bool = True,
) -> pd.DataFrame:
    """
    Fill missing values in a time-series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
    method : str
        'ffill' (forward fill) or 'interpolate' (linear interpolation).
    drop_remaining : bool
        If True, drop rows that still contain NaN after filling
        (typically leading rows that have no prior data).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()

    if method == "ffill":
        df = df.ffill()
    elif method == "interpolate":
        df = df.interpolate(method="linear")
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'ffill' or 'interpolate'.")

    if drop_remaining:
        df = df.dropna()

    return df


# ── Outlier Detection & Treatment ────────────────────────────────────

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Flag outliers using the IQR method.

    Parameters
    ----------
    series : pd.Series
        Numeric series to check.
    k : float
        Multiplier for the IQR (1.5 = standard, 3.0 = extreme).

    Returns
    -------
    pd.Series
        Boolean mask — True where the value is an outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def winsorize_series(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """
    Winsorize a series by clipping at the given percentiles.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    lower, upper : float
        Percentile bounds (0–1).

    Returns
    -------
    pd.Series
        Clipped series.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def winsorize_dataframe(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Winsorize selected (or all numeric) columns in a DataFrame.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        df[col] = winsorize_series(df[col], lower, upper)
    return df


# ── Scaling ──────────────────────────────────────────────────────────

def scale_features(
    df: pd.DataFrame,
    method: str = "standard",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler]:
    """
    Scale numeric features using scikit-learn.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    method : str
        'standard' (zero mean, unit variance) or 'minmax' (0-1 range).
    columns : list[str] or None
        Columns to scale. None = all numeric columns.

    Returns
    -------
    tuple[pd.DataFrame, scaler]
        Scaled DataFrame and the fitted scaler object.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'standard' or 'minmax'.")

    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler
