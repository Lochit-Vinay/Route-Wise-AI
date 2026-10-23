"""
Tests for Sensitivity Analysis Module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sensitivity import (
    weight_sensitivity_analysis,
    constraint_sensitivity,
    summarize_weight_sensitivity,
)
from src.optimizer import OptimizationWeights


@pytest.fixture
def sensitivity_sample_routes():
    """Routes with clear trade-offs to observe switching during sensitivity analysis."""
    return pd.DataFrame({
        "shipment_id": ["SH-SENS"] * 3,
        "route_id": ["ROUTE-CHEAP", "ROUTE-FAST", "ROUTE-SAFE"],
        "intermediate_hubs": ["Direct", "Singapore", "Dubai"],
        "shipping_cost": [1000.0, 5000.0, 3000.0],
        "transit_days": [25.0, 5.0, 15.0],
        "delay_probability": [0.30, 0.20, 0.05],
        "congestion_risk": [0.30, 0.20, 0.05],
        "geopolitical_risk": [0.20, 0.15, 0.05],
    })


class TestWeightSensitivity:
    """Tests for weight sensitivity analysis."""

    def test_weight_sensitivity_generation(self, sensitivity_sample_routes):
        results = weight_sensitivity_analysis(
            sensitivity_sample_routes,
            max_budget=6000.0,
            max_transit_days=30.0,
            max_delay_probability=0.35,
            weight_steps=6,
        )

        assert not results.empty
        assert "cost_weight" in results.columns
        assert "time_weight" in results.columns
        assert "risk_weight" in results.columns
        assert "recommended_route" in results.columns
        assert "optimization_score" in results.columns

        # Verify weights sum to 1 in each row
        weight_sums = results["cost_weight"] + results["time_weight"] + results["risk_weight"]
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-5)

    def test_weight_sensitivity_switches_routes(self, sensitivity_sample_routes):
        """High cost weight selects cheap route; high time weight selects fast route."""
        results = weight_sensitivity_analysis(
            sensitivity_sample_routes,
            max_budget=6000.0,
            max_transit_days=30.0,
            max_delay_probability=0.35,
            weight_steps=11,
        )

        summary = summarize_weight_sensitivity(results)
        # Should see multiple different routes selected depending on weights
        assert summary["unique_routes"] >= 2
        assert summary["most_frequent"] in ["ROUTE-CHEAP", "ROUTE-FAST", "ROUTE-SAFE"]
        assert 0.0 < summary["stability"] <= 1.0

    def test_weight_sensitivity_no_feasible(self, sensitivity_sample_routes):
        """If constraints are impossible, empty DataFrame is returned safely."""
        results = weight_sensitivity_analysis(
            sensitivity_sample_routes,
            max_budget=500.0,  # Below minimum 1000.0
            max_transit_days=30.0,
            max_delay_probability=0.35,
        )
        assert results.empty
        summary = summarize_weight_sensitivity(results)
        assert summary["unique_routes"] == 0
        assert summary["most_frequent"] is None


class TestConstraintSensitivity:
    """Tests for constraint sensitivity analysis."""

    def test_constraint_sensitivity_all_dimensions(self, sensitivity_sample_routes):
        res = constraint_sensitivity(
            sensitivity_sample_routes,
            base_budget=3000.0,
            base_transit=15.0,
            base_delay=0.20,
            steps=5,
        )

        assert "budget" in res
        assert "transit" in res
        assert "delay" in res

        for dim in ["budget", "transit", "delay"]:
            df = res[dim]
            assert len(df) == 5
            assert "multiplier" in df.columns
            assert "feasible_count" in df.columns
            assert "recommended_route" in df.columns

    def test_tight_budget_drops_feasible_count(self, sensitivity_sample_routes):
        res = constraint_sensitivity(
            sensitivity_sample_routes,
            base_budget=3000.0,
            base_transit=30.0,
            base_delay=0.50,
            budget_range=(0.2, 2.0),
            steps=5,
        )
        b_df = res["budget"]
        # Lowest multiplier (0.2 * 3000 = 600) should have 0 feasible routes
        assert b_df.iloc[0]["feasible_count"] == 0
        # Highest multiplier (2.0 * 3000 = 6000) should have 3 feasible routes
        assert b_df.iloc[-1]["feasible_count"] == 3
