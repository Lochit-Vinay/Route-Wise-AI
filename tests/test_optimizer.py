"""
Tests for the Multi-Objective Optimization Engine.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimizer import (
    OptimizationWeights,
    normalize_metrics,
    compute_weighted_scores,
    optimize_routes,
    get_route_scores,
    _safe_normalize,
    compute_pareto_frontier,
    get_priority_weights,
)
from src.preprocessing import preprocess_routes


@pytest.fixture
def sample_routes():
    """Create sample route data for testing."""
    return pd.DataFrame({
        "shipment_id": ["SH-001"] * 3,
        "route_id": ["R1", "R2", "R3"],
        "intermediate_hubs": ["Hub A", "Hub B", "Hub C"],
        "shipping_cost": [1000.0, 3000.0, 2000.0],
        "transit_days": [10.0, 5.0, 8.0],
        "delay_probability": [0.1, 0.3, 0.2],
        "congestion_risk": [0.15, 0.35, 0.25],
        "geopolitical_risk": [0.1, 0.2, 0.15],
    })


@pytest.fixture
def processed_routes(sample_routes):
    """Create preprocessed route data."""
    return preprocess_routes(sample_routes)


class TestOptimizationWeights:
    """Tests for weight validation."""

    def test_default_weights(self):
        w = OptimizationWeights()
        assert w.cost == 0.40
        assert w.time == 0.35
        assert w.risk == 0.25

    def test_custom_weights(self):
        w = OptimizationWeights(cost=0.5, time=0.3, risk=0.2)
        assert w.cost == 0.5
        assert w.time == 0.3
        assert w.risk == 0.2

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            OptimizationWeights(cost=0.5, time=0.5, risk=0.5)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            OptimizationWeights(cost=-0.1, time=0.6, risk=0.5)

    def test_weight_above_one_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            OptimizationWeights(cost=1.5, time=0.0, risk=-0.5)

    def test_non_numeric_weight_rejected(self):
        with pytest.raises(TypeError, match="must be numeric"):
            OptimizationWeights(cost="high", time=0.3, risk=0.2)

    def test_extreme_weights(self):
        # All weight on cost
        w = OptimizationWeights(cost=1.0, time=0.0, risk=0.0)
        assert w.cost == 1.0
        assert w.time == 0.0

    def test_weights_to_dict(self):
        w = OptimizationWeights()
        d = w.to_dict()
        assert d == {"cost": 0.40, "time": 0.35, "risk": 0.25}


class TestNormalization:
    """Tests for min-max normalization."""

    def test_normalization_range(self, processed_routes):
        result = normalize_metrics(processed_routes)
        df = result.normalized_values

        for col in ["cost_normalized", "time_normalized", "risk_normalized"]:
            assert df[col].min() >= 0.0, f"{col} has values below 0"
            assert df[col].max() <= 1.0, f"{col} has values above 1"

    def test_min_value_normalizes_to_zero(self, processed_routes):
        result = normalize_metrics(processed_routes)
        df = result.normalized_values

        # The route with lowest cost should have cost_normalized = 0
        min_cost_idx = processed_routes["shipping_cost"].idxmin()
        assert df.loc[min_cost_idx, "cost_normalized"] == pytest.approx(0.0, abs=1e-6)

    def test_max_value_normalizes_to_one(self, processed_routes):
        result = normalize_metrics(processed_routes)
        df = result.normalized_values

        max_cost_idx = processed_routes["shipping_cost"].idxmax()
        assert df.loc[max_cost_idx, "cost_normalized"] == pytest.approx(1.0, abs=1e-6)

    def test_division_by_zero_handling(self):
        """When all values are the same, normalization should return 0."""
        identical_routes = pd.DataFrame({
            "shipment_id": ["SH-001"] * 3,
            "route_id": ["R1", "R2", "R3"],
            "intermediate_hubs": ["A", "B", "C"],
            "shipping_cost": [5000.0, 5000.0, 5000.0],
            "transit_days": [10.0, 10.0, 10.0],
            "delay_probability": [0.2, 0.2, 0.2],
            "congestion_risk": [0.3, 0.3, 0.3],
            "geopolitical_risk": [0.1, 0.1, 0.1],
        })
        processed = preprocess_routes(identical_routes)
        result = normalize_metrics(processed)
        df = result.normalized_values

        assert (df["cost_normalized"] == 0.0).all()
        assert (df["time_normalized"] == 0.0).all()
        assert (df["risk_normalized"] == 0.0).all()

    def test_safe_normalize_function(self):
        series = pd.Series([1.0, 2.0, 3.0])
        result = _safe_normalize(series, 1.0, 3.0)
        expected = pd.Series([0.0, 0.5, 1.0])
        pd.testing.assert_series_equal(result, expected, atol=1e-6)

    def test_safe_normalize_zero_range(self):
        series = pd.Series([5.0, 5.0, 5.0])
        result = _safe_normalize(series, 5.0, 5.0)
        assert (result == 0.0).all()

    def test_normalization_bounds_stored(self, processed_routes):
        result = normalize_metrics(processed_routes)
        assert result.cost_min <= result.cost_max
        assert result.time_min <= result.time_max
        assert result.risk_min <= result.risk_max


class TestWeightedScoring:
    """Tests for weighted score calculation."""

    def test_score_calculation(self, processed_routes):
        result = normalize_metrics(processed_routes)
        weights = OptimizationWeights(cost=0.4, time=0.35, risk=0.25)
        scored = compute_weighted_scores(result.normalized_values, weights)

        for _, row in scored.iterrows():
            expected = (
                weights.cost * row["cost_normalized"]
                + weights.time * row["time_normalized"]
                + weights.risk * row["risk_normalized"]
            )
            assert row["optimization_score"] == pytest.approx(expected, abs=1e-5)

    def test_lower_score_is_better(self, processed_routes):
        """Route with rank 1 should have the lowest score."""
        result = normalize_metrics(processed_routes)
        scored = compute_weighted_scores(result.normalized_values, OptimizationWeights())
        assert scored.iloc[0]["rank"] == 1
        assert scored.iloc[0]["optimization_score"] <= scored.iloc[-1]["optimization_score"]

    def test_ranking_consistency(self, processed_routes):
        """Ranks should be sequential and start from 1."""
        scored, _ = optimize_routes(processed_routes)
        ranks = sorted(scored["rank"].tolist())
        assert ranks[0] == 1

    def test_all_cost_weight(self):
        """With all weight on cost, cheapest route should rank first."""
        routes = pd.DataFrame({
            "shipment_id": ["SH-001"] * 3,
            "route_id": ["R1", "R2", "R3"],
            "intermediate_hubs": ["A", "B", "C"],
            "shipping_cost": [1000.0, 5000.0, 3000.0],
            "transit_days": [20.0, 5.0, 10.0],
            "delay_probability": [0.3, 0.05, 0.15],
            "congestion_risk": [0.4, 0.1, 0.2],
            "geopolitical_risk": [0.2, 0.05, 0.1],
        })
        processed = preprocess_routes(routes)
        weights = OptimizationWeights(cost=1.0, time=0.0, risk=0.0)
        scored, _ = optimize_routes(processed, weights)
        assert scored.iloc[0]["route_id"] == "R1"  # Cheapest

    def test_all_time_weight(self):
        """With all weight on time, fastest route should rank first."""
        routes = pd.DataFrame({
            "shipment_id": ["SH-001"] * 3,
            "route_id": ["R1", "R2", "R3"],
            "intermediate_hubs": ["A", "B", "C"],
            "shipping_cost": [1000.0, 5000.0, 3000.0],
            "transit_days": [20.0, 5.0, 10.0],
            "delay_probability": [0.3, 0.05, 0.15],
            "congestion_risk": [0.4, 0.1, 0.2],
            "geopolitical_risk": [0.2, 0.05, 0.1],
        })
        processed = preprocess_routes(routes)
        weights = OptimizationWeights(cost=0.0, time=1.0, risk=0.0)
        scored, _ = optimize_routes(processed, weights)
        assert scored.iloc[0]["route_id"] == "R2"  # Fastest

    def test_empty_routes(self):
        """Optimizer should handle empty DataFrame gracefully."""
        empty = pd.DataFrame(columns=[
            "shipment_id", "route_id", "intermediate_hubs",
            "shipping_cost", "transit_days", "delay_probability",
            "congestion_risk", "geopolitical_risk",
        ])
        scored, _ = optimize_routes(empty)
        assert scored.empty

    def test_single_route(self):
        """Single route should rank 1 with score 0 (all normalized to 0)."""
        routes = pd.DataFrame({
            "shipment_id": ["SH-001"],
            "route_id": ["R1"],
            "intermediate_hubs": ["Direct"],
            "shipping_cost": [5000.0],
            "transit_days": [10.0],
            "delay_probability": [0.1],
            "congestion_risk": [0.2],
            "geopolitical_risk": [0.1],
        })
        processed = preprocess_routes(routes)
        scored, _ = optimize_routes(processed)
        assert len(scored) == 1
        assert scored.iloc[0]["rank"] == 1
        assert scored.iloc[0]["optimization_score"] == pytest.approx(0.0, abs=1e-6)


class TestRouteScores:
    """Tests for detailed score breakdown."""

    def test_get_route_scores(self, processed_routes):
        scores = get_route_scores(processed_routes)
        assert len(scores) == 3
        assert scores[0].rank == 1
        assert scores[0].total_score <= scores[-1].total_score

    def test_score_attributes(self, processed_routes):
        scores = get_route_scores(processed_routes)
        for score in scores:
            assert 0 <= score.norm_cost <= 1
            assert 0 <= score.norm_time <= 1
            assert 0 <= score.norm_risk <= 1
            assert score.total_score >= 0


class TestParetoOptimality:
    """Tests for Pareto efficiency and non-domination analysis."""

    def test_pareto_frontier_identifies_dominated_route(self):
        """Route dominated across cost, time, and risk must not be Pareto optimal."""
        routes = pd.DataFrame({
            "route_id": ["R_eff_1", "R_eff_2", "R_dom"],
            "shipping_cost": [1000.0, 2000.0, 2500.0],
            "transit_days": [10.0, 5.0, 12.0],
            "composite_risk": [0.20, 0.10, 0.25],
        }).set_index("route_id")
        frontier = compute_pareto_frontier(routes)
        assert bool(frontier["R_eff_1"]) is True
        assert bool(frontier["R_eff_2"]) is True
        # R_dom is strictly worse than R_eff_1 in cost, time, and risk
        assert bool(frontier["R_dom"]) is False

    def test_linear_scalarization_invariant(self, processed_routes):
        """Invariant: Optimal route under strictly positive weights must be Pareto optimal."""
        weights = OptimizationWeights(cost=0.5, time=0.3, risk=0.2)
        scored, _ = optimize_routes(processed_routes, weights)
        best_route = scored.iloc[0]
        assert bool(best_route["is_pareto_optimal"]) is True

    def test_priority_weights_profiles(self):
        """Test preset weight generators for operational shipment priorities."""
        crit = get_priority_weights("Critical")
        assert crit.time > crit.cost
        assert crit.risk > crit.cost
        assert pytest.approx(crit.cost + crit.time + crit.risk) == 1.0

        low = get_priority_weights("Low")
        assert low.cost > low.time
        assert low.cost > low.risk
        assert pytest.approx(low.cost + low.time + low.risk) == 1.0

        # Case insensitivity
        med = get_priority_weights("medium")
        assert med.cost == 0.40

        # Fallback for unrecognized priority
        unknown = get_priority_weights("SuperUrgent")
        assert unknown.cost == 0.40

