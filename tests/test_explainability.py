"""
Tests for Explainability Module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.explainability import (
    generate_score_comparison_table,
    explain_weight_impact,
    generate_trade_off_summary,
    format_constraint_status,
)
from src.optimizer import OptimizationWeights
from src.constraints import evaluate_shipment_constraints


class TestExplainability:
    """Tests for explainability, comparisons, and transparency."""

    @pytest.fixture
    def scored_sample(self):
        return pd.DataFrame([
            {
                "route_id": "R1",
                "shipping_cost": 2000.0,
                "transit_days": 15.0,
                "delay_probability": 0.10,
                "composite_risk": 0.12,
                "cost_normalized": 0.0,
                "time_normalized": 0.5,
                "risk_normalized": 0.2,
                "weighted_cost": 0.0,
                "weighted_time": 0.175,
                "weighted_risk": 0.05,
                "optimization_score": 0.225,
                "rank": 1,
                "is_pareto_optimal": True,
            },
            {
                "route_id": "R2",
                "shipping_cost": 3500.0,
                "transit_days": 10.0,
                "delay_probability": 0.08,
                "composite_risk": 0.09,
                "cost_normalized": 1.0,
                "time_normalized": 0.0,
                "risk_normalized": 0.0,
                "weighted_cost": 0.40,
                "weighted_time": 0.0,
                "weighted_risk": 0.0,
                "optimization_score": 0.40,
                "rank": 2,
                "is_pareto_optimal": False,
            },
        ])

    def test_score_comparison_table(self, scored_sample):
        weights = OptimizationWeights()
        table = generate_score_comparison_table(scored_sample, weights)

        assert not table.empty
        assert "Route ID" in table.columns
        assert "Cost ($)" in table.columns
        assert "Transit (days)" in table.columns
        assert "Score" in table.columns
        assert "Pareto Optimal" in table.columns

    def test_explain_weight_impact(self, scored_sample):
        weights = OptimizationWeights(cost=0.40, time=0.35, risk=0.25)
        impact = explain_weight_impact(scored_sample.iloc[0], weights)

        assert "cost_pct" in impact
        assert "time_pct" in impact
        assert "risk_pct" in impact
        # Sum of percentages should be ~100%
        total_pct = impact["cost_pct"] + impact["time_pct"] + impact["risk_pct"]
        assert abs(total_pct - 100.0) < 1.0

    def test_generate_trade_off_summary(self, scored_sample):
        summary = generate_trade_off_summary(scored_sample, "R1")

        assert "Trade-off Analysis for R1" in summary
        # R2 is faster and lower risk
        assert "faster" in summary.lower() or "risk" in summary.lower()

    def test_format_constraint_status(self):
        sample_routes = pd.DataFrame([
            {
                "route_id": "R1",
                "shipping_cost": 2000.0,
                "transit_days": 10.0,
                "delay_probability": 0.10,
            },
            {
                "route_id": "R2",
                "shipping_cost": 5000.0,
                "transit_days": 20.0,
                "delay_probability": 0.40,
            },
        ])

        _, results = evaluate_shipment_constraints(
            sample_routes, max_budget=3000.0, max_transit_days=15.0, max_delay_probability=0.25
        )

        status_df = format_constraint_status(results)
        assert len(status_df) == 2
        assert "Route ID" in status_df.columns
        assert "Feasible" in status_df.columns
        assert "✅" in status_df.iloc[0]["Feasible"]
        assert "❌" in status_df.iloc[1]["Feasible"]

    def test_format_constraint_status_with_soft_warnings(self):
        sample_routes = pd.DataFrame([
            {
                "route_id": "R_close",
                "shipping_cost": 2950.0,  # within 5% of 3000
                "transit_days": 10.0,
                "delay_probability": 0.10,
            },
        ])
        _, results = evaluate_shipment_constraints(
            sample_routes, max_budget=3000.0, max_transit_days=15.0, max_delay_probability=0.25
        )
        status_df = format_constraint_status(results)
        assert "Advisories" in status_df.columns
        adv = status_df.iloc[0]["Advisories"].lower()
        assert "slack" in adv or "tight" in adv

