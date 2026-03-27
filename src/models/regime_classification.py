"""
Regime Classification module for Phase 5.
Contains rule-based and KMeans clustering approaches for identifying business cycles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_rule_based_regimes(df: pd.DataFrame, growth_col: str, inflation_col: str) -> pd.DataFrame:
    """
    Classifies regimes based on Growth and Inflation.
    Expansion: Growth > 0, Inflation < 0 (or median)
    Overheating: Growth > 0, Inflation > 0
    Stagflation: Growth < 0, Inflation > 0
    Contraction: Growth < 0, Inflation < 0
    
    Uses median to center the data so it represents relative positive/negative.
    """
    df_out = df.copy()
    
    if growth_col not in df.columns or inflation_col not in df.columns:
        df_out['regime_rule_based'] = "Unknown"
        return df_out
        
    g_median = df[growth_col].median()
    i_median = df[inflation_col].median()
    
    # Growth+, Inflation-
    cond_exp = (df[growth_col] >= g_median) & (df[inflation_col] < i_median)
    # Growth+, Inflation+
    cond_ovr = (df[growth_col] >= g_median) & (df[inflation_col] >= i_median)
    # Growth-, Inflation+
    cond_stg = (df[growth_col] < g_median) & (df[inflation_col] >= i_median)
    # Growth-, Inflation-
    cond_con = (df[growth_col] < g_median) & (df[inflation_col] < i_median)
    
    df_out['regime_rule_based'] = np.select(
        [cond_exp, cond_ovr, cond_stg, cond_con],
        ['Expansion', 'Overheating', 'Stagflation', 'Contraction'],
        default='Unknown'
    )
    
    return df_out

def compute_kmeans_regimes(df: pd.DataFrame, macro_cols: list[str], n_clusters: int = 4) -> pd.DataFrame:
    """
    Classifies regimes using KMeans clustering on normalized macro features.
    """
    df_out = df.copy()
    valid_cols = [c for c in macro_cols if c in df.columns]
    
    if not valid_cols:
        df_out['regime_kmeans'] = "Unknown"
        return df_out
        
    # Dropna to avoid scaling issues, but we only predict on valid rows
    data_to_fit = df[valid_cols].dropna()
    if data_to_fit.empty:
        df_out['regime_kmeans'] = "Unknown"
        return df_out
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_fit)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled_data)
    
    # Create a mapping dictionary based on chronological order or simple indices
    regime_map = {i: f"Cluster {i+1}" for i in range(n_clusters)}
    
    # Reindex labels to match original DataFrame
    df_out['regime_kmeans'] = pd.Series(labels, index=data_to_fit.index).map(regime_map)
    df_out['regime_kmeans'] = df_out['regime_kmeans'].fillna("Unknown")
    
    return df_out

def plot_regime_timeline(df: pd.DataFrame, regime_col: str, title: str = "Regime Timeline", ax=None):
    """
    Plots a color-coded timeline of regimes.
    """
    if regime_col not in df.columns:
        return None
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))
    else:
        fig = ax.figure
        
    regimes = df[regime_col].unique()
    regimes = [r for r in regimes if r != "Unknown"]
    color_map = {reg: plt.cm.tab10(i) for i, reg in enumerate(regimes)}
    color_map["Unknown"] = "gray"
    
    # Plot bars of timeline
    for i, (date, row) in enumerate(df.iterrows()):
        ax.axvspan(date, date + pd.Timedelta(days=30), color=color_map.get(row[regime_col], "gray"), alpha=0.6)
        
    ax.set_title(title)
    ax.set_yticks([]) # Hide y-axis
    ax.set_xlabel("Date")
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[r], label=r, alpha=0.6) for r in regimes]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    return fig

def get_regime_performance(df: pd.DataFrame, regime_col: str, sectors: list[str], ret_type: str = "excess_ret") -> pd.DataFrame:
    """
    Returns average return of sectors grouped by regime.
    """
    ret_cols = [f"{s}_{ret_type}" for s in sectors if f"{s}_{ret_type}" in df.columns]
    
    if not ret_cols or regime_col not in df.columns:
        return pd.DataFrame()
        
    return df.groupby(regime_col)[ret_cols].mean()
