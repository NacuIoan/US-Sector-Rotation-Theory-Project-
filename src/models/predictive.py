"""
Predictive Modeling module for Phase 7.
Trains Logistic Regression models to predict 1-month outperformance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def prepare_classification_data(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares data for classification by dropping NA values for the required columns.
    Returns (X, y).
    """
    cols = [target_col] + feature_cols
    valid_cols = [c for c in cols if c in df.columns]
    
    if len(valid_cols) < len(cols):
        raise ValueError(f"Missing columns for prediction: {set(cols) - set(valid_cols)}")
        
    data = df[valid_cols].dropna()
    if data.empty:
        raise ValueError("No data left after dropping NAs.")
        
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y

def train_logistic_regression(X: pd.DataFrame, y: pd.Series):
    """
    Trains a Logistic Regression model on scaled features.
    Returns (scaler, model).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_scaled, y)
    
    return scaler, model

def evaluate_model_tscv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Evaluates the model using TimeSeriesSplit cross-validation to prevent data leakage.
    Returns average accuracy, precision, recall, and f1 score.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    
    X_arr = X.values
    y_arr = y.values
    
    for train_index, test_index in tscv.split(X_arr):
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        
        # Scale locally to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # We need both classes to train safely
        if len(np.unique(y_train)) < 2:
            continue
            
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        
    if not metrics["accuracy"]:
        return {"error": "Not enough valid splits for evaluation."}
        
    return {k: np.mean(v) for k, v in metrics.items()}

def get_feature_importance(model: LogisticRegression, feature_cols: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame of coefficients sorted by absolute magnitude.
    """
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0],
        "Abs_Coefficient": np.abs(model.coef_[0])
    })
    return importance.sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

def run_all_sector_predictions(df: pd.DataFrame, sectors: list[str], feature_cols: list[str]) -> dict:
    """
    Runs predictive modeling pipeline for all sectors.
    Returns a dictionary mapping sector to models and metrics.
    """
    results = {}
    for sector in sectors:
        target_col = f"{sector}_outperform_1m"
        if target_col in df.columns:
            try:
                X, y = prepare_classification_data(df, target_col, feature_cols)
                
                # Full train for feature importance and dashboard
                scaler, model = train_logistic_regression(X, y)
                
                # Cross-validation for realistic evaluation
                cv_metrics = evaluate_model_tscv(X, y)
                
                importance_df = get_feature_importance(model, feature_cols)
                
                # Last timeframe prediction (dummy approach for dashboard demonstration)
                last_row = X.iloc[-1:]
                last_row_scaled = scaler.transform(last_row)
                pred_prob = model.predict_proba(last_row_scaled)[0][1]
                
                results[sector] = {
                    "model": model,
                    "scaler": scaler,
                    "cv_metrics": cv_metrics,
                    "feature_importance": importance_df,
                    "latest_probability": float(pred_prob)
                }
            except Exception as e:
                results[sector] = {"error": str(e)}
    return results
