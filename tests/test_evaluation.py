"""
Tests for Evaluation Module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    evaluate_baseline_cheapest,
    evaluate_baseline_fastest,
    evaluate_baseline_lowest_risk,
    evaluate_weighted_optimization,
    run_full_evaluation,
    evaluation_to_dataframe,
    EvaluationResult,
)
from src.optimizer import OptimizationWeights


@pytest.fixture
def eval_test_data():
    """Create a mini dataset with 3 shipments and contrasting routes."""
    shipments = pd.DataFrame([
        {
            "shipment_id": "SH-E1",
            "origin": "Shanghai",
            "destination": "Rotterdam",
            "priority": "Medium",
            "max_budget": 5000.0,
            "max_transit_days": 25.0,
            "max_delay_probability": 0.35,
        },
        {
            "shipment_id": "SH-E2",
            "origin": "Los Angeles",
            "destination": "Tokyo",
            "priority": "Critical",
            "max_budget": 8000.0,
            "max_transit_days": 18.0,
            "max_delay_probability": 0.25,
        },
        {
            "shipment_id": "SH-E3_INFEASIBLE",
            "origin": "Dubai",
            "destination": "Hamburg",
            "priority": "High",
            "max_budget": 100.0,  # Unattainable budget
            "max_transit_days": 2.0,
            "max_delay_probability": 0.01,
        },
    ])

    routes = pd.DataFrame([
        # SH-E1 routes
        {
            "shipment_id": "SH-E1",
            "route_id": "SH-E1-R1",  # Cheapest
            "intermediate_hubs": "Direct",
            "shipping_cost": 2000.0,
            "transit_days": 24.0,
            "delay_probability": 0.30,
            "congestion_risk": 0.25,
            "geopolitical_risk": 0.10,
        },
        {
            "shipment_id": "SH-E1",
            "route_id": "SH-E1-R2",  # Fastest
            "intermediate_hubs": "Singapore",
            "shipping_cost": 4500.0,
            "transit_days": 10.0,
            "delay_probability": 0.15,
            "congestion_risk": 0.10,
            "geopolitical_risk": 0.10,
        },
        {
            "shipment_id": "SH-E1",
            "route_id": "SH-E1-R3",  # Balanced / Safe
            "intermediate_hubs": "Dubai",
            "shipping_cost": 2800.0,
            "transit_days": 14.0,
            "delay_probability": 0.12,
            "congestion_risk": 0.10,
            "geopolitical_risk": 0.08,
        },
        # SH-E2 routes
        {
            "shipment_id": "SH-E2",
            "route_id": "SH-E2-R1",  # Cheap
            "intermediate_hubs": "Direct",
            "shipping_cost": 3000.0,
            "transit_days": 16.0,
            "delay_probability": 0.20,
            "congestion_risk": 0.20,
            "geopolitical_risk": 0.10,
        },
        {
            "shipment_id": "SH-E2",
            "route_id": "SH-E2-R2",  # Fast & low risk
            "intermediate_hubs": "Direct",
            "shipping_cost": 6000.0,
            "transit_days": 8.0,
            "delay_probability": 0.08,
            "congestion_risk": 0.08,
            "geopolitical_risk": 0.05,
        },
        # SH-E3 routes (all will violate constraints)
        {
            "shipment_id": "SH-E3_INFEASIBLE",
            "route_id": "SH-E3-R1",
            "intermediate_hubs": "Direct",
            "shipping_cost": 3000.0,
            "transit_days": 12.0,
            "delay_probability": 0.18,
            "congestion_risk": 0.15,
            "geopolitical_risk": 0.12,
        },
    ])

    return shipments, routes


class TestBaselineStrategies:
    """Tests for each baseline strategy and comparison."""

    def test_cheapest_route_strategy(self, eval_test_data):
        shipments, routes = eval_test_data
        result = evaluate_baseline_cheapest(shipments, routes)

        assert isinstance(result, EvaluationResult)
        assert result.strategy_name == "Cheapest Route"
        assert result.shipments_evaluated == 3
        assert result.shipments_with_feasible == 2
        assert result.shipments_no_feasible == 1
        assert result.selected_routes["SH-E1"] == "SH-E1-R1"  # 2000 vs 4500, 2800
        assert result.selected_routes["SH-E2"] == "SH-E2-R1"  # 3000 vs 6000
        assert result.selected_routes["SH-E3_INFEASIBLE"] is None

    def test_fastest_route_strategy(self, eval_test_data):
        shipments, routes = eval_test_data
        result = evaluate_baseline_fastest(shipments, routes)

        assert result.strategy_name == "Fastest Route"
        assert result.selected_routes["SH-E1"] == "SH-E1-R2"  # 10 days vs 24, 14
        assert result.selected_routes["SH-E2"] == "SH-E2-R2"  # 8 days vs 16

    def test_lowest_risk_route_strategy(self, eval_test_data):
        shipments, routes = eval_test_data
        result = evaluate_baseline_lowest_risk(shipments, routes)

        assert result.strategy_name == "Lowest-Risk Route"
        assert result.selected_routes["SH-E1"] == "SH-E1-R3"  # 0.12 delay vs 0.30, 0.15
        assert result.selected_routes["SH-E2"] == "SH-E2-R2"  # 0.08 delay vs 0.20

    def test_weighted_optimization_strategy(self, eval_test_data):
        shipments, routes = eval_test_data
        weights = OptimizationWeights(cost=0.40, time=0.35, risk=0.25)
        result = evaluate_weighted_optimization(shipments, routes, weights)

        assert result.strategy_name == "Weighted Optimization"
        assert result.shipments_with_feasible == 2
        assert result.shipments_no_feasible == 1
        assert result.avg_optimization_score is not None

    def test_run_full_evaluation(self, eval_test_data):
        shipments, routes = eval_test_data
        results = run_full_evaluation(shipments, routes)

        assert "Weighted Optimization" in results
        assert "Cheapest Route" in results
        assert "Fastest Route" in results
        assert "Lowest-Risk Route" in results

        df = evaluation_to_dataframe(results)
        assert len(df) == 4
        assert "Strategy" in df.columns
        assert "Avg Cost ($)" in df.columns
        assert "Avg Transit (days)" in df.columns
        assert "Avg Delay Prob" in df.columns
        assert "Avg Opt Score" in df.columns

    def test_empty_dataset_evaluation(self):
        empty_s = pd.DataFrame(columns=["shipment_id", "max_budget", "max_transit_days", "max_delay_probability"])
        empty_r = pd.DataFrame(columns=["shipment_id", "route_id", "shipping_cost", "transit_days", "delay_probability", "congestion_risk", "geopolitical_risk"])

        res = evaluate_weighted_optimization(empty_s, empty_r)
        assert res.shipments_evaluated == 0
        assert res.shipments_with_feasible == 0
        assert res.avg_cost == 0
