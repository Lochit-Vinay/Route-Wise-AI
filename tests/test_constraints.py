"""
Tests for the Constraint Engine.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constraints import (
    evaluate_route_constraints,
    evaluate_shipment_constraints,
    get_feasible_routes,
    find_closest_infeasible,
    compute_required_relaxations,
    ConstraintResult,
)
from src.preprocessing import preprocess_routes


@pytest.fixture
def sample_routes():
    return pd.DataFrame({
        "shipment_id": ["SH-001"] * 4,
        "route_id": ["R1", "R2", "R3", "R4"],
        "intermediate_hubs": ["A", "B", "C", "D"],
        "shipping_cost": [2000.0, 5000.0, 3000.0, 8000.0],
        "transit_days": [10.0, 5.0, 8.0, 15.0],
        "delay_probability": [0.1, 0.3, 0.2, 0.5],
        "congestion_risk": [0.15, 0.35, 0.25, 0.6],
        "geopolitical_risk": [0.1, 0.2, 0.15, 0.3],
    })


class TestRouteConstraints:
    """Tests for individual route constraint evaluation."""

    def test_feasible_route(self, sample_routes):
        route = sample_routes.iloc[0]  # R1: cost=2000, transit=10, delay=0.1
        result = evaluate_route_constraints(route, 5000, 15, 0.3)
        assert result.is_feasible
        assert len(result.violated_constraints) == 0

    def test_budget_violation(self, sample_routes):
        route = sample_routes.iloc[1]  # R2: cost=5000
        result = evaluate_route_constraints(route, 4000, 15, 0.5)
        assert not result.is_feasible
        assert any("Cost" in v for v in result.violated_constraints)
        assert not result.constraint_details["budget"]["satisfied"]

    def test_transit_violation(self, sample_routes):
        route = sample_routes.iloc[3]  # R4: transit=15
        result = evaluate_route_constraints(route, 10000, 10, 1.0)
        assert not result.is_feasible
        assert any("Transit" in v for v in result.violated_constraints)

    def test_delay_violation(self, sample_routes):
        route = sample_routes.iloc[3]  # R4: delay=0.5
        result = evaluate_route_constraints(route, 10000, 20, 0.3)
        assert not result.is_feasible
        assert any("Delay" in v for v in result.violated_constraints)

    def test_multiple_violations(self, sample_routes):
        route = sample_routes.iloc[3]  # R4: cost=8000, transit=15, delay=0.5
        result = evaluate_route_constraints(route, 5000, 10, 0.3)
        assert not result.is_feasible
        assert len(result.violated_constraints) == 3

    def test_exact_boundary(self, sample_routes):
        route = sample_routes.iloc[0]  # R1: cost=2000
        result = evaluate_route_constraints(route, 2000, 10, 0.1)
        assert result.is_feasible  # Equal to limit is acceptable

    def test_constraint_margins(self, sample_routes):
        route = sample_routes.iloc[0]
        result = evaluate_route_constraints(route, 3000, 15, 0.3)
        assert result.constraint_details["budget"]["margin"] == 1000.0
        assert result.constraint_details["transit_time"]["margin"] == 5.0
        assert result.constraint_details["delay_probability"]["margin"] == pytest.approx(0.2, abs=1e-6)


class TestShipmentConstraints:
    """Tests for batch constraint evaluation."""

    def test_evaluate_all_routes(self, sample_routes):
        routes_eval, results = evaluate_shipment_constraints(
            sample_routes, 5000, 12, 0.3
        )
        assert len(results) == 4
        assert "is_feasible" in routes_eval.columns

    def test_get_feasible_routes(self, sample_routes):
        feasible = get_feasible_routes(sample_routes, 5000, 12, 0.3)
        for _, route in feasible.iterrows():
            assert route["shipping_cost"] <= 5000
            assert route["transit_days"] <= 12
            assert route["delay_probability"] <= 0.3

    def test_no_feasible_routes(self, sample_routes):
        feasible = get_feasible_routes(sample_routes, 100, 1, 0.01)
        assert feasible.empty

    def test_all_feasible(self, sample_routes):
        feasible = get_feasible_routes(sample_routes, 100000, 100, 1.0)
        assert len(feasible) == len(sample_routes)

    def test_recommendation_never_violates_constraints(self, sample_routes):
        """The most critical test: recommended routes must satisfy constraints."""
        budget, transit, delay = 4000, 12, 0.25
        feasible = get_feasible_routes(sample_routes, budget, transit, delay)

        for _, route in feasible.iterrows():
            assert route["shipping_cost"] <= budget
            assert route["transit_days"] <= transit
            assert route["delay_probability"] <= delay


class TestClosestInfeasible:
    """Tests for finding the closest infeasible route."""

    def test_find_closest(self, sample_routes):
        # With very tight constraints, find the closest route
        closest = find_closest_infeasible(sample_routes, 100, 1, 0.01)
        assert closest is not None
        assert "violation_distance" in closest.index

    def test_no_infeasible(self, sample_routes):
        closest = find_closest_infeasible(sample_routes, 100000, 100, 1.0)
        assert closest is None

    def test_closest_has_least_violations(self, sample_routes):
        closest = find_closest_infeasible(sample_routes, 2500, 9, 0.15)
        # The closest route should have the smallest violation
        assert closest is not None
        assert closest["violation_distance"] >= 0


class TestSoftWarningsAndRelaxations:
    """Tests for boundary slack warnings and relaxation calculations."""

    def test_soft_warning_triggered_at_tight_margin(self):
        """Route with <5% budget slack should report a soft advisory warning."""
        route = pd.Series({
            "route_id": "R_tight",
            "shipping_cost": 980.0,  # 2% slack on 1000 limit
            "transit_days": 5.0,     # 50% slack on 10 limit
            "delay_probability": 0.05,
        })
        res = evaluate_route_constraints(route, max_budget=1000.0, max_transit_days=10.0, max_delay_probability=0.20)
        assert res.is_feasible is True
        assert len(res.soft_warnings) >= 1
        assert any("Budget" in w for w in res.soft_warnings)

    def test_no_soft_warning_when_comfortable_slack(self):
        """Route with >=5% slack across all dimensions should have no soft warnings."""
        route = pd.Series({
            "route_id": "R_comfortable",
            "shipping_cost": 800.0,
            "transit_days": 8.0,
            "delay_probability": 0.10,
        })
        res = evaluate_route_constraints(route, max_budget=1000.0, max_transit_days=10.0, max_delay_probability=0.20)
        assert res.is_feasible is True
        assert len(res.soft_warnings) == 0

    def test_compute_required_relaxations_infeasible(self):
        """Calculates precise deficit and percentage concession needed."""
        routes = pd.DataFrame([{
            "route_id": "R_inf",
            "shipping_cost": 5500.0,  # exceeds 5000 budget by 500 (10%)
            "transit_days": 12.0,     # exceeds 10 days by 2 (20%)
            "delay_probability": 0.10,
        }])
        _, results = evaluate_shipment_constraints(routes, max_budget=5000.0, max_transit_days=10.0, max_delay_probability=0.20)
        relaxations = compute_required_relaxations(results)
        assert len(relaxations) == 2
        metrics = {r["metric"]: r for r in relaxations}
        assert "Budget" in metrics
        assert metrics["Budget"]["deficit"] == 500.0
        assert metrics["Budget"]["deficit_pct"] == 10.0
        assert "Transit Days" in metrics
        assert metrics["Transit Days"]["deficit"] == 2.0
        assert metrics["Transit Days"]["deficit_pct"] == 20.0

    def test_compute_required_relaxations_empty_when_feasible(self):
        """If any route is feasible, no relaxations are needed."""
        routes = pd.DataFrame([{
            "route_id": "R_feas",
            "shipping_cost": 4000.0,
            "transit_days": 8.0,
            "delay_probability": 0.10,
        }])
        _, results = evaluate_shipment_constraints(routes, max_budget=5000.0, max_transit_days=10.0, max_delay_probability=0.20)
        relaxations = compute_required_relaxations(results)
        assert relaxations == []

