"""
Tests for Preprocessing Module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import preprocess_routes, compute_hub_count, _classify_risk


class TestPreprocessing:
    """Tests for route preprocessing and feature engineering."""

    def test_preprocess_routes_adds_composite_columns(self):
        routes = pd.DataFrame({
            "shipment_id": ["S1"],
            "route_id": ["R1"],
            "shipping_cost": ["1500.50"],  # string to test coercion
            "transit_days": [12.0],
            "delay_probability": [0.20],
            "congestion_risk": [0.30],
            "geopolitical_risk": [0.10],
        })

        processed = preprocess_routes(routes)

        assert "composite_risk" in processed.columns
        assert "reliability_score" in processed.columns
        assert "risk_level" in processed.columns
        assert pd.api.types.is_numeric_dtype(processed["shipping_cost"])

        # Composite risk formula: 0.50 * delay + 0.30 * cong + 0.20 * geo
        expected_risk = round(0.50 * 0.20 + 0.30 * 0.30 + 0.20 * 0.10, 4)
        assert processed.iloc[0]["composite_risk"] == expected_risk
        assert processed.iloc[0]["reliability_score"] == round(1.0 - expected_risk, 4)

    def test_classify_risk_thresholds(self):
        assert _classify_risk(0.10) == "Low"
        assert _classify_risk(0.15) == "Low"
        assert _classify_risk(0.25) == "Medium"
        assert _classify_risk(0.30) == "Medium"
        assert _classify_risk(0.40) == "High"
        assert _classify_risk(0.45) == "High"
        assert _classify_risk(0.60) == "Critical"

    def test_compute_hub_count(self):
        assert compute_hub_count("Direct") == 0
        assert compute_hub_count("direct") == 0
        assert compute_hub_count(None) == 0
        assert compute_hub_count("Singapore") == 1
        assert compute_hub_count("Singapore → Dubai") == 2
        assert compute_hub_count("Ningbo → Port Klang → Salalah") == 3
