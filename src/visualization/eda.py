"""
Exploratory Data Analysis (EDA) module for Phase 4.
Contains functions for plotting and summarizing data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_descriptive_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Returns descriptive statistics for the specified columns.
    """
    cols_to_use = [c for c in columns if c in df.columns]
    return df[cols_to_use].describe().T


def get_groupby_stats(df: pd.DataFrame, group_col: str, target_cols: list[str]) -> pd.DataFrame:
    """
    Returns mean and std of target_cols grouped by group_col.
    """
    cols_to_use = [group_col] + [c for c in target_cols if c in df.columns]
    return df[cols_to_use].groupby(group_col).agg(["mean", "std", "count"])


def plot_return_distributions(df: pd.DataFrame, sectors: list[str], ret_type: str = "excess_ret", ax=None):
    """
    Plots a boxplot (or KDE) of returns for the given sectors.
    """
    cols = [f"{s}_{ret_type}" for s in sectors if f"{s}_{ret_type}" in df.columns]
    
    if df[cols].empty:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    sns.boxplot(data=df[cols], ax=ax)
    ax.set_title(f"Return Distributions ({ret_type.replace('_', ' ').title()})")
    ax.set_ylabel("Return")
    ax.set_xticklabels(sectors, rotation=45)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], ax=None):
    """
    Plots a correlation heatmap for the specified columns.
    """
    cols_to_use = [c for c in columns if c in df.columns]
    if len(cols_to_use) < 2:
        return None

    corr = df[cols_to_use].corr()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap")
    return fig


def plot_time_series(df: pd.DataFrame, columns: list[str], title: str = "Time Series", ax=None):
    """
    Plots time series lines for the specified columns.
    Assumes df has a DatetimeIndex or 'Date' is the index.
    """
    cols_to_use = [c for c in columns if c in df.columns]
    
    if not cols_to_use:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    df[cols_to_use].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig


def create_excess_return_summary(df: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    """
    Creates a summary table comparing each sector's excess return vs SPY.
    (Excess return means return minus SPY return, so its mean should be > 0 
    if it outperforms SPY on average).
    """
    cols = [f"{s}_excess_ret" for s in sectors if f"{s}_excess_ret" in df.columns]
    
    if not cols:
        return pd.DataFrame()

    summary = pd.DataFrame({
        "Mean Excess Return (Monthly)": df[cols].mean(),
        "Median Excess Return": df[cols].median(),
        "Volatility (Std)": df[cols].std(),
        "Win Rate vs SPY": (df[cols] > 0).mean()
    })
    
    summary.index = [c.replace("_excess_ret", "") for c in summary.index]
    return summary.sort_values("Mean Excess Return (Monthly)", ascending=False)
