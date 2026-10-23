"""
Tests for Data Generator and Data Loader Modules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import (
    generate_dataset,
    load_dataset,
    LOGISTICS_HUBS,
    PRIORITIES,
    _estimate_distance_category,
    _generate_intermediate_hubs,
)
from src.data_loader import get_data, get_merged_data, _validate_data


class TestDataGenerator:
    """Tests for synthetic dataset generator."""

    def test_dataset_generation_shapes_and_bounds(self, tmp_path):
        out_dir = str(tmp_path / "data_test")
        shipments, routes = generate_dataset(num_shipments=35, seed=123, output_dir=out_dir)

        # 30-50 shipments requirement
        assert len(shipments) == 35
        # 3-5 routes per shipment requirement
        routes_per_ship = routes.groupby("shipment_id").size()
        assert routes_per_ship.min() >= 3
        assert routes_per_ship.max() <= 5

        # Check required columns
        for col in ["shipment_id", "origin", "destination", "priority", "max_budget", "max_transit_days", "max_delay_probability"]:
            assert col in shipments.columns

        for col in ["shipment_id", "route_id", "intermediate_hubs", "shipping_cost", "transit_days", "delay_probability", "congestion_risk", "geopolitical_risk"]:
            assert col in routes.columns

        # Verify value sanity
        assert (routes["shipping_cost"] > 0).all()
        assert (routes["transit_days"] > 0).all()
        assert ((routes["delay_probability"] >= 0) & (routes["delay_probability"] <= 1)).all()
        assert ((routes["congestion_risk"] >= 0) & (routes["congestion_risk"] <= 1)).all()
        assert ((routes["geopolitical_risk"] >= 0) & (routes["geopolitical_risk"] <= 1)).all()

    def test_reproducibility_with_seed(self, tmp_path):
        dir1 = str(tmp_path / "d1")
        dir2 = str(tmp_path / "d2")

        s1, r1 = generate_dataset(num_shipments=10, seed=999, output_dir=dir1)
        s2, r2 = generate_dataset(num_shipments=10, seed=999, output_dir=dir2)

        pd.testing.assert_frame_equal(s1, s2)
        pd.testing.assert_frame_equal(r1, r2)

    def test_distance_category_logic(self):
        # East Asia to East Asia -> short
        assert _estimate_distance_category("Shanghai", "Tokyo") == "short"
        # East Asia to Southeast Asia -> medium
        assert _estimate_distance_category("Shanghai", "Singapore") == "medium"
        # East Asia to Europe -> long
        assert _estimate_distance_category("Shanghai", "Rotterdam") == "long"

    def test_intermediate_hubs_generation(self):
        rng = np.random.default_rng(42)
        hubs = _generate_intermediate_hubs("Shanghai", "Rotterdam", rng)
        assert isinstance(hubs, str)
        assert len(hubs) > 0


class TestDataLoader:
    """Tests for data loading and validation."""

    def test_get_data_and_merged_data(self):
        s, r = get_data()
        assert len(s) >= 30
        assert len(r) >= 90

        merged = get_merged_data()
        assert len(merged) == len(r)
        assert "max_budget" in merged.columns
        assert "shipping_cost" in merged.columns

    def test_validate_data_catches_invalid_schemas(self):
        valid_s = pd.DataFrame({
            "shipment_id": ["S1"], "origin": ["A"], "destination": ["B"],
            "priority": ["High"], "max_budget": [1000.0],
            "max_transit_days": [10.0], "max_delay_probability": [0.2],
        })
        valid_r = pd.DataFrame({
            "shipment_id": ["S1"], "route_id": ["R1"], "intermediate_hubs": ["Direct"],
            "shipping_cost": [500.0], "transit_days": [5.0], "delay_probability": [0.1],
            "congestion_risk": [0.1], "geopolitical_risk": [0.1],
        })

        # Should pass
        _validate_data(valid_s, valid_r)

        # Missing column in shipments
        invalid_s = valid_s.drop(columns=["max_budget"])
        with pytest.raises(ValueError, match="Missing required column in shipments"):
            _validate_data(invalid_s, valid_r)

        # Negative cost in routes
        negative_r = valid_r.copy()
        negative_r["shipping_cost"] = -10.0
        with pytest.raises(ValueError, match="Negative values found"):
            _validate_data(valid_s, negative_r)

    def test_validate_data_rejects_out_of_range_and_non_finite_values(self):
        valid_s = pd.DataFrame({
            "shipment_id": ["S1"], "origin": ["A"], "destination": ["B"],
            "priority": ["High"], "max_budget": [1000.0],
            "max_transit_days": [10.0], "max_delay_probability": [0.2],
        })
        valid_r = pd.DataFrame({
            "shipment_id": ["S1"], "route_id": ["R1"], "intermediate_hubs": ["Direct"],
            "shipping_cost": [500.0], "transit_days": [5.0], "delay_probability": [0.1],
            "congestion_risk": [0.1], "geopolitical_risk": [0.1],
        })

        invalid_probability = valid_r.copy()
        invalid_probability.loc[0, "delay_probability"] = 1.2
        with pytest.raises(ValueError, match="between 0 and 1"):
            _validate_data(valid_s, invalid_probability)

        non_finite_cost = valid_r.copy()
        non_finite_cost.loc[0, "shipping_cost"] = np.inf
        with pytest.raises(ValueError, match="Non-finite"):
            _validate_data(valid_s, non_finite_cost)

    def test_validate_data_rejects_orphaned_and_duplicate_identifiers(self):
        valid_s = pd.DataFrame({
            "shipment_id": ["S1"], "origin": ["A"], "destination": ["B"],
            "priority": ["High"], "max_budget": [1000.0],
            "max_transit_days": [10.0], "max_delay_probability": [0.2],
        })
        valid_r = pd.DataFrame({
            "shipment_id": ["S1"], "route_id": ["R1"], "intermediate_hubs": ["Direct"],
            "shipping_cost": [500.0], "transit_days": [5.0], "delay_probability": [0.1],
            "congestion_risk": [0.1], "geopolitical_risk": [0.1],
        })

        orphan = valid_r.copy()
        orphan.loc[0, "shipment_id"] = "UNKNOWN"
        with pytest.raises(ValueError, match="unknown shipments"):
            _validate_data(valid_s, orphan)

        duplicate = pd.concat([valid_r, valid_r], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate route_id"):
            _validate_data(valid_s, duplicate)
