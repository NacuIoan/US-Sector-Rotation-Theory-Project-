"""
Econometric Modeling module for Phase 6.
Runs OLS regressions of sector returns against macroeconomic features.
"""

import pandas as pd
import statsmodels.api as sm

def run_sector_regression(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """
    Runs an OLS regression for a single sector against specified macro features.
    Drops NA values for the required columns before fitting.
    Returns the fitted statsmodels RegressionResultsWrapper.
    """
    cols = [target_col] + feature_cols
    valid_cols = [c for c in cols if c in df.columns]
    
    if len(valid_cols) < len(cols):
        missing = set(cols) - set(valid_cols)
        raise ValueError(f"Missing columns for regression: {missing}")
        
    data = df[valid_cols].dropna()
    if data.empty:
        raise ValueError("No data left after dropping NAs.")
        
    X = data[feature_cols]
    y = data[target_col]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    return model

def get_regression_summary(model) -> pd.DataFrame:
    """
    Extracts key metrics from a statsmodels OLS result.
    Returns a DataFrame with Coefficients, P-Values, and R-squared.
    """
    summary = pd.DataFrame({
        "Coefficient": model.params,
        "P-Value": model.pvalues,
    })
    
    # Add an indicator for statistical significance (p < 0.05)
    summary["Significant"] = summary["P-Value"] < 0.05
    
    return summary

def get_model_stats(model) -> dict:
    """
    Returns high-level statistics of the model.
    """
    return {
        "R-squared": model.rsquared,
        "Adj. R-squared": model.rsquared_adj,
        "F-statistic p-value": model.f_pvalue,
        "Observations": int(model.nobs)
    }

def run_all_sector_regressions(df: pd.DataFrame, sectors: list[str], feature_cols: list[str], ret_type: str = "excess_ret") -> dict:
    """
    Runs regression for all sectors and returns a dictionary mapping sector to summary objects.
    """
    results = {}
    for sector in sectors:
        target_col = f"{sector}_{ret_type}"
        if target_col in df.columns:
            try:
                model = run_sector_regression(df, target_col, feature_cols)
                results[sector] = {
                    "model": model,
                    "summary_df": get_regression_summary(model),
                    "stats": get_model_stats(model)
                }
            except Exception as e:
                results[sector] = {"error": str(e)}
    return results
