"""
Dashboard for Phase 8.
Streamlit application integrating all analytical models and visualizations.
Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.data.data_loader import load_processed_dataset
from src.utils.config import SECTOR_ETFS
from src.visualization.eda import (
    get_descriptive_stats, plot_return_distributions, 
    plot_correlation_heatmap, plot_time_series, create_excess_return_summary
)
from src.models.regime_classification import (
    compute_rule_based_regimes, compute_kmeans_regimes,
    plot_regime_timeline, get_regime_performance
)
from src.models.econometrics import run_all_sector_regressions
from src.models.predictive import run_all_sector_predictions

st.set_page_config(page_title="Sector Rotation Theory", layout="wide")

@st.cache_data
def load_data():
    try:
        df = load_processed_dataset()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.title("Configuration")

# Date range
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.sidebar.slider(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM"
)

# Filter Data by Date
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
filtered_df = df.loc[mask]

# Sectors
available_sectors = [s for s in SECTOR_ETFS if f"{s}_excess_ret" in df.columns]
selected_sectors = st.sidebar.multiselect("Select Sectors", available_sectors, default=available_sectors[:5])

# Macro features
potential_macros = [c for c in df.columns if "_yoy" in c or "_mom" in c or "_lag" in c]
selected_macros = st.sidebar.multiselect("Select Macro Indicators", potential_macros, default=potential_macros[:3])

st.title("Sector Rotation & Macro Indicators Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["EDA & View", "Regime Explorer", "Econometrics", "Predictive Models"])

# --- Tab 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    st.write(f"Analyzed Rows: {filtered_df.shape[0]}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time Series (Sectors)")
        if selected_sectors:
            fig1 = plot_time_series(filtered_df, [f"{s}_excess_ret" for s in selected_sectors], "Sector Excess Returns")
            st.pyplot(fig1)
    
    with col2:
        st.subheader("Time Series (Macros)")
        if selected_macros:
            fig2 = plot_time_series(filtered_df, selected_macros, "Macro Indicators")
            st.pyplot(fig2)
            
    st.subheader("Correlation Heatmap")
    if selected_sectors and selected_macros:
        cols_to_corr = [f"{s}_excess_ret" for s in selected_sectors] + selected_macros
        fig3 = plot_correlation_heatmap(filtered_df, cols_to_corr)
        st.pyplot(fig3)
        
    st.subheader("Return Distributions")
    if selected_sectors:
        fig4 = plot_return_distributions(filtered_df, selected_sectors)
        st.pyplot(fig4)
        
    st.subheader("Sector Summary against SPY")
    summary_df = create_excess_return_summary(filtered_df, selected_sectors)
    st.dataframe(summary_df)

# --- Tab 2: Regimes ---
with tab2:
    st.header("Business Cycle & Regimes")
    
    growth_options = [c for c in df.columns if "mom" in c or "INDPRO" in c]
    inflation_options = [c for c in df.columns if "yoy" in c or "CPI" in c]
    
    c1, c2 = st.columns(2)
    growth_proxy = c1.selectbox("Growth Proxy", growth_options)
    inflation_proxy = c2.selectbox("Inflation Proxy", inflation_options)
    
    regime_df = compute_rule_based_regimes(filtered_df, growth_proxy, inflation_proxy)
    regime_df = compute_kmeans_regimes(regime_df, selected_macros, n_clusters=4)
    
    regime_type = st.radio("Regime Type", ["Rule-Based", "KMeans Clustering"])
    reg_col = "regime_rule_based" if regime_type == "Rule-Based" else "regime_kmeans"
    
    st.subheader("Regime Timeline")
    fig = plot_regime_timeline(regime_df, reg_col)
    st.pyplot(fig)
    
    st.subheader("Sector Performance by Regime (Excess Return)")
    perf_df = get_regime_performance(regime_df, reg_col, selected_sectors)
    st.dataframe(perf_df.style.background_gradient(cmap="RdYlGn", axis=None))

# --- Tab 3: Econometrics ---
with tab3:
    st.header("OLS Regressions")
    st.write("Sector Excess Returns ~ Macro Indicators")
    
    if st.button("Run Regressions", key="ols_btn"):
        if not selected_sectors or not selected_macros:
            st.warning("Please select at least one sector and one macro indicator.")
        else:
            with st.spinner("Running OLS models..."):
                results = run_all_sector_regressions(filtered_df, selected_sectors, selected_macros)
                
                for sector, res in results.items():
                    st.subheader(f"{sector} OLS Results")
                    if "error" in res:
                        st.error(f"Error: {res['error']}")
                    else:
                        st.write(res["stats"])
                        st.dataframe(res["summary_df"].style.apply(lambda x: ['font-weight: bold' if v else '' for v in x], subset=['Significant']))

# --- Tab 4: Predictive Models ---
with tab4:
    st.header("Logistic Regression")
    st.write("Predicting 1-Month Outperformance (`outperform_1m`) vs SPY")
    
    if st.button("Train Models", key="pred_btn"):
        if not selected_sectors or not selected_macros:
            st.warning("Please select at least one sector and one macro indicator.")
        else:
            with st.spinner("Training predictive models with TimeSeries CV..."):
                # Use entire df to allow enough rows for CV
                pred_results = run_all_sector_predictions(df, selected_sectors, selected_macros)
                
                for sector, res in pred_results.items():
                    st.subheader(f"{sector} Prediction Metrics")
                    if "error" in res:
                        st.error(f"Error: {res['error']}")
                    elif "cv_metrics" in res and "error" in res["cv_metrics"]:
                         st.error(f"CV Error for {sector}: {res['cv_metrics']['error']}")
                    else:
                        st.write(f"Latest probability of outperforming next month: {res.get('latest_probability', 0):.1%}")
                        st.write("Average CV Metrics:")
                        st.json(res["cv_metrics"])
                        st.write("Feature Importance:")
                        st.dataframe(res["feature_importance"])
