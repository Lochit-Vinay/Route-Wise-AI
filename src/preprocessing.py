"""
Preprocessing utilities for RouteWise AI.

Handles data cleaning, type casting, and derived column creation.
"""

import pandas as pd
import numpy as np
from typing import Optional


def preprocess_routes(routes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess route data.

    - Ensure correct data types.
    - Compute composite risk score.
    - Add reliability indicator.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Raw routes DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed routes DataFrame.
    """
    df = routes_df.copy()

    # Ensure numeric types
    numeric_cols = ["shipping_cost", "transit_days", "delay_probability",
                    "congestion_risk", "geopolitical_risk"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any NaN values with column median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Composite risk score: weighted combination of delay, congestion, geo risk
    df["composite_risk"] = (
        0.50 * df["delay_probability"]
        + 0.30 * df["congestion_risk"]
        + 0.20 * df["geopolitical_risk"]
    )
    df["composite_risk"] = df["composite_risk"].round(4)

    # Reliability indicator (inverse of composite risk)
    df["reliability_score"] = (1 - df["composite_risk"]).round(4)

    # Risk classification
    df["risk_level"] = df["composite_risk"].apply(_classify_risk)

    return df


def _classify_risk(risk_score: float) -> str:
    """Classify risk level based on composite risk score."""
    if risk_score <= 0.15:
        return "Low"
    elif risk_score <= 0.30:
        return "Medium"
    elif risk_score <= 0.45:
        return "High"
    else:
        return "Critical"


def compute_hub_count(hubs_str: str) -> int:
    """Count the number of intermediate hubs."""
    if pd.isna(hubs_str) or hubs_str.strip().lower() == "direct":
        return 0
    return len(hubs_str.split("→"))
