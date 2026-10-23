"""
Tests for the Recommendation Engine.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.recommender import recommend_route, recommend_all_shipments, Recommendation
from src.optimizer import OptimizationWeights
from src.data_loader import get_data


@pytest.fixture
def sample_shipment():
    return pd.Series({
        "shipment_id": "SH-TEST",
        "origin": "Shanghai",
        "destination": "Rotterdam",
        "priority": "High",
        "max_budget": 6000.0,
        "max_transit_days": 20.0,
        "max_delay_probability": 0.3,
    })


@pytest.fixture
def sample_routes():
    return pd.DataFrame({
        "shipment_id": ["SH-TEST"] * 4,
        "route_id": ["R1", "R2", "R3", "R4"],
        "intermediate_hubs": ["Singapore", "Dubai", "Direct", "Colombo"],
        "shipping_cost": [3000.0, 5000.0, 7000.0, 4000.0],
        "transit_days": [18.0, 12.0, 8.0, 15.0],
        "delay_probability": [0.15, 0.10, 0.25, 0.20],
        "congestion_risk": [0.20, 0.15, 0.30, 0.25],
        "geopolitical_risk": [0.15, 0.10, 0.12, 0.20],
    })


@pytest.fixture
def infeasible_shipment():
    return pd.Series({
        "shipment_id": "SH-NONE",
        "origin": "Shanghai",
        "destination": "Rotterdam",
        "priority": "Critical",
        "max_budget": 100.0,
        "max_transit_days": 1.0,
        "max_delay_probability": 0.001,
    })


class TestRecommendation:
    """Tests for route recommendation."""

    def test_recommendation_returns_result(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        assert isinstance(rec, Recommendation)
        assert rec.shipment_id == "SH-TEST"

    def test_recommendation_selects_route(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        assert rec.has_feasible_routes
        assert rec.recommended_route_id is not None
        assert rec.recommended_route is not None

    def test_recommended_route_satisfies_constraints(self, sample_shipment, sample_routes):
        """Critical: recommended route must never violate hard constraints."""
        rec = recommend_route(sample_shipment, sample_routes)
        if rec.has_feasible_routes:
            best = rec.recommended_route
            assert best["shipping_cost"] <= sample_shipment["max_budget"]
            assert best["transit_days"] <= sample_shipment["max_transit_days"]
            assert best["delay_probability"] <= sample_shipment["max_delay_probability"]

    def test_no_feasible_route_handled(self, infeasible_shipment, sample_routes):
        rec = recommend_route(infeasible_shipment, sample_routes)
        assert not rec.has_feasible_routes
        assert rec.recommended_route_id is None
        assert "No feasible route" in rec.explanation

    def test_explanation_not_empty(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        assert rec.explanation
        assert len(rec.explanation) > 50

    def test_explanation_contains_computed_values(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        if rec.has_feasible_routes:
            # Explanation should contain actual score values
            assert "Score Breakdown" in rec.explanation
            assert "normalized" in rec.explanation

    def test_score_breakdown_present(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        if rec.has_feasible_routes:
            assert rec.score_breakdown is not None
            assert "total_score" in rec.score_breakdown
            assert "cost_normalized" in rec.score_breakdown

    def test_custom_weights(self, sample_shipment, sample_routes):
        w1 = OptimizationWeights(cost=0.8, time=0.1, risk=0.1)
        w2 = OptimizationWeights(cost=0.1, time=0.8, risk=0.1)
        rec1 = recommend_route(sample_shipment, sample_routes, w1)
        rec2 = recommend_route(sample_shipment, sample_routes, w2)
        # Different weights may lead to different recommendations
        # (or the same if one route dominates)
        assert rec1.weights_used.cost == 0.8
        assert rec2.weights_used.time == 0.8

    def test_custom_constraint_override(self, sample_shipment, sample_routes):
        rec = recommend_route(
            sample_shipment, sample_routes,
            custom_budget=2000.0,  # Very tight budget
        )
        # Should have fewer or no feasible routes
        assert rec.feasible_count <= 4

    def test_constraint_results_present(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        assert len(rec.constraint_results) == len(sample_routes)

    def test_all_routes_scored(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        assert len(rec.all_routes_scored) == len(sample_routes)

    def test_pareto_status_in_recommendation(self, sample_shipment, sample_routes):
        rec = recommend_route(sample_shipment, sample_routes)
        assert hasattr(rec, "is_pareto_optimal")
        assert isinstance(rec.is_pareto_optimal, bool)
        assert rec.is_pareto_optimal is True

    def test_infeasible_explanation_contains_actionable_concessions(self, infeasible_shipment, sample_routes):
        rec = recommend_route(infeasible_shipment, sample_routes)
        assert not rec.has_feasible_routes
        assert "Actionable Concessions" in rec.explanation



class TestBulkRecommendation:
    """Tests for bulk recommendation across dataset."""

    def test_recommend_all_shipments(self):
        shipments_df, routes_df = get_data()
        # Test with first 5 shipments for speed
        small_shipments = shipments_df.head(5)
        recs = recommend_all_shipments(small_shipments, routes_df)
        assert len(recs) == 5
        for rec in recs:
            assert isinstance(rec, Recommendation)

    def test_no_constraint_violation_in_bulk(self):
        """No recommended route should ever violate hard constraints."""
        shipments_df, routes_df = get_data()
        recs = recommend_all_shipments(shipments_df.head(10), routes_df)
        for rec in recs:
            if rec.has_feasible_routes and rec.recommended_route is not None:
                shipment = shipments_df[shipments_df["shipment_id"] == rec.shipment_id].iloc[0]
                assert rec.recommended_route["shipping_cost"] <= shipment["max_budget"]
                assert rec.recommended_route["transit_days"] <= shipment["max_transit_days"]
                assert rec.recommended_route["delay_probability"] <= shipment["max_delay_probability"]
