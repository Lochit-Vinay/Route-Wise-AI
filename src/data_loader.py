"""
Data Loader for RouteWise AI.

Provides centralized data loading with validation and caching
for the Streamlit application.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
from src.data_generator import generate_dataset, load_dataset


def get_data(data_dir: str = "data", regenerate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load or generate the dataset.

    Parameters
    ----------
    data_dir : str
        Directory for CSV files.
    regenerate : bool
        If True, regenerate the dataset even if files exist.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        shipments_df, routes_df
    """
    shipments_path = os.path.join(data_dir, "shipments.csv")
    routes_path = os.path.join(data_dir, "routes.csv")

    if regenerate or not os.path.exists(shipments_path) or not os.path.exists(routes_path):
        shipments_df, routes_df = generate_dataset(output_dir=data_dir)
    else:
        shipments_df, routes_df = load_dataset(data_dir)

    # Validate data integrity
    _validate_data(shipments_df, routes_df)

    return shipments_df, routes_df


def get_merged_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load and merge shipment and route data.

    Returns a single DataFrame with shipment constraints joined
    to each route alternative.
    """
    shipments_df, routes_df = get_data(data_dir)
    merged = routes_df.merge(shipments_df, on="shipment_id", how="left")
    return merged


def _validate_data(shipments_df: pd.DataFrame, routes_df: pd.DataFrame) -> None:
    """Validate dataset integrity."""
    required_shipment_cols = [
        "shipment_id", "origin", "destination", "priority",
        "max_budget", "max_transit_days", "max_delay_probability",
    ]
    required_route_cols = [
        "shipment_id", "route_id", "intermediate_hubs",
        "shipping_cost", "transit_days", "delay_probability",
        "congestion_risk", "geopolitical_risk",
    ]

    for col in required_shipment_cols:
        if col not in shipments_df.columns:
            raise ValueError(f"Missing required column in shipments: {col}")

    for col in required_route_cols:
        if col not in routes_df.columns:
            raise ValueError(f"Missing required column in routes: {col}")

    # IDs must be present and unique at their respective entity levels.
    if shipments_df["shipment_id"].isna().any() or routes_df[["shipment_id", "route_id"]].isna().any().any():
        raise ValueError("Shipment and route identifiers must not contain missing values")
    if shipments_df["shipment_id"].duplicated().any():
        raise ValueError("Duplicate shipment_id values found in shipments")
    if routes_df["route_id"].duplicated().any():
        raise ValueError("Duplicate route_id values found in routes")

    # Every shipment must have routes, and every route must reference a known shipment.
    shipment_ids = set(shipments_df["shipment_id"].unique())
    shipment_ids_with_routes = set(routes_df["shipment_id"].unique())
    orphaned = shipment_ids_with_routes - shipment_ids
    if orphaned:
        raise ValueError(f"Routes reference unknown shipments: {orphaned}")
    missing = shipment_ids - shipment_ids_with_routes
    if missing:
        raise ValueError(f"Shipments without any routes: {missing}")

    # Validate numeric values for finiteness, sign, and probability bounds.
    route_nonnegative = ["shipping_cost", "transit_days", "delay_probability", "congestion_risk", "geopolitical_risk"]
    for col in route_nonnegative:
        values = pd.to_numeric(routes_df[col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite or non-numeric values found in routes column: {col}")
        if (values < 0).any():
            raise ValueError(f"Negative values found in routes column: {col}")

    for col in ["delay_probability", "congestion_risk", "geopolitical_risk"]:
        if (routes_df[col] > 1).any():
            raise ValueError(f"Probability/risk values must be between 0 and 1 in routes column: {col}")

    shipment_limits = ["max_budget", "max_transit_days", "max_delay_probability"]
    for col in shipment_limits:
        values = pd.to_numeric(shipments_df[col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite or non-numeric values found in shipments column: {col}")
        if (values < 0).any():
            raise ValueError(f"Negative values found in shipments column: {col}")
    if (shipments_df["max_delay_probability"] > 1).any():
        raise ValueError("max_delay_probability must be between 0 and 1")
