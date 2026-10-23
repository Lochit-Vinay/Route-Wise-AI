"""
Synthetic Dataset Generator for RouteWise AI.

Generates realistic shipment and route data for multi-objective
route optimization evaluation.

All data is synthetic and does not represent actual logistics
or historical shipping data. Generated using a fixed random seed
for full reproducibility.

Assumptions:
-----------
1. Shipments originate and terminate at major global logistics hubs.
2. Shipping costs range from $800 to $15,000 depending on distance,
   mode, and route complexity.
3. Transit times range from 3 to 45 days depending on route.
4. Delay probabilities range from 0.02 to 0.55.
5. Congestion and geopolitical risks are on a 0-1 scale.
6. Each shipment has 3-5 alternative routes with meaningful variation.
7. Priority levels: Critical, High, Medium, Low.
8. Budget and time constraints are set to create a mix of feasible
   and infeasible scenarios, including shipments with no feasible route.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import os

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Major logistics hubs for realistic origin-destination pairs
LOGISTICS_HUBS = {
    "Shanghai": {"region": "East Asia", "geo_risk_base": 0.15},
    "Singapore": {"region": "Southeast Asia", "geo_risk_base": 0.08},
    "Rotterdam": {"region": "Europe", "geo_risk_base": 0.05},
    "Dubai": {"region": "Middle East", "geo_risk_base": 0.20},
    "Los Angeles": {"region": "North America", "geo_risk_base": 0.06},
    "Hamburg": {"region": "Europe", "geo_risk_base": 0.05},
    "Mumbai": {"region": "South Asia", "geo_risk_base": 0.18},
    "Busan": {"region": "East Asia", "geo_risk_base": 0.10},
    "Tokyo": {"region": "East Asia", "geo_risk_base": 0.07},
    "Sydney": {"region": "Oceania", "geo_risk_base": 0.04},
    "New York": {"region": "North America", "geo_risk_base": 0.06},
    "Santos": {"region": "South America", "geo_risk_base": 0.16},
    "Jeddah": {"region": "Middle East", "geo_risk_base": 0.22},
    "Colombo": {"region": "South Asia", "geo_risk_base": 0.14},
    "Cape Town": {"region": "Africa", "geo_risk_base": 0.19},
}

# Intermediate hub pools by region
INTERMEDIATE_HUBS = {
    "East Asia": ["Ningbo", "Qingdao", "Kaohsiung", "Hong Kong", "Xiamen"],
    "Southeast Asia": ["Port Klang", "Tanjung Pelepas", "Laem Chabang", "Ho Chi Minh City", "Manila"],
    "Europe": ["Antwerp", "Felixstowe", "Piraeus", "Barcelona", "Genoa", "Gdansk"],
    "Middle East": ["Jebel Ali", "Salalah", "King Abdullah Port", "Sohar"],
    "North America": ["Long Beach", "Savannah", "Charleston", "Oakland", "Houston"],
    "South Asia": ["Mundra", "Nhava Sheva", "Chittagong", "Karachi"],
    "Oceania": ["Melbourne", "Brisbane", "Auckland", "Tauranga"],
    "South America": ["Buenos Aires", "Callao", "Cartagena", "Manzanillo"],
    "Africa": ["Durban", "Mombasa", "Tangier Med", "Lagos"],
}

# Priority definitions
PRIORITIES = ["Critical", "High", "Medium", "Low"]
PRIORITY_WEIGHTS = [0.10, 0.25, 0.40, 0.25]

# Route distance approximation categories (determines base cost/time)
def _estimate_distance_category(origin: str, destination: str) -> str:
    """Estimate relative distance between two hubs."""
    hub_data = LOGISTICS_HUBS
    o_region = hub_data[origin]["region"]
    d_region = hub_data[destination]["region"]

    if o_region == d_region:
        return "short"
    
    nearby_pairs = {
        ("East Asia", "Southeast Asia"), ("Southeast Asia", "East Asia"),
        ("Europe", "Europe"), ("South Asia", "Southeast Asia"),
        ("Southeast Asia", "South Asia"), ("Middle East", "South Asia"),
        ("South Asia", "Middle East"), ("Europe", "Middle East"),
        ("Middle East", "Europe"), ("North America", "South America"),
        ("South America", "North America"),
    }
    if (o_region, d_region) in nearby_pairs:
        return "medium"
    
    return "long"


def _generate_intermediate_hubs(
    origin: str,
    destination: str,
    rng: np.random.Generator,
    num_hubs_range: Tuple[int, int] = (1, 3),
) -> str:
    """Generate a realistic set of intermediate hubs for a route."""
    o_region = LOGISTICS_HUBS[origin]["region"]
    d_region = LOGISTICS_HUBS[destination]["region"]

    # Collect candidate hubs from relevant regions
    candidate_pools = []
    for region in set([o_region, d_region]):
        if region in INTERMEDIATE_HUBS:
            candidate_pools.extend(INTERMEDIATE_HUBS[region])

    # Add hubs from connecting regions for long routes
    if _estimate_distance_category(origin, destination) == "long":
        connecting_regions = ["Southeast Asia", "Middle East", "Europe"]
        for region in connecting_regions:
            if region not in [o_region, d_region] and region in INTERMEDIATE_HUBS:
                candidate_pools.extend(INTERMEDIATE_HUBS[region][:2])

    # Remove origin and destination from candidates
    candidate_pools = [h for h in candidate_pools if h not in [origin, destination]]
    
    if not candidate_pools:
        return "Direct"

    num_hubs = rng.integers(num_hubs_range[0], num_hubs_range[1] + 1)
    num_hubs = min(num_hubs, len(candidate_pools))
    
    selected = rng.choice(candidate_pools, size=num_hubs, replace=False)
    return " → ".join(selected)


def generate_dataset(
    num_shipments: int = 40,
    seed: int = RANDOM_SEED,
    output_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic shipment and route dataset.

    Parameters
    ----------
    num_shipments : int
        Number of shipments to generate (30-50 recommended).
    seed : int
        Random seed for reproducibility.
    output_dir : str
        Directory to save CSV files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        shipments_df, routes_df
    """
    rng = np.random.default_rng(seed)
    hub_names = list(LOGISTICS_HUBS.keys())

    shipments_records = []
    routes_records = []

    for i in range(num_shipments):
        shipment_id = f"SH-{i+1:03d}"

        # Select origin and destination (must differ)
        origin, destination = rng.choice(hub_names, size=2, replace=False)

        # Assign priority
        priority = rng.choice(PRIORITIES, p=PRIORITY_WEIGHTS)

        # Determine distance category for realistic parameters
        dist_cat = _estimate_distance_category(origin, destination)

        # Base parameters by distance category
        if dist_cat == "short":
            base_cost_range = (800, 3500)
            base_time_range = (3, 12)
            base_delay_range = (0.03, 0.20)
        elif dist_cat == "medium":
            base_cost_range = (2000, 7000)
            base_time_range = (8, 22)
            base_delay_range = (0.05, 0.30)
        else:  # long
            base_cost_range = (4000, 15000)
            base_time_range = (15, 45)
            base_delay_range = (0.08, 0.55)

        # Set constraints based on priority
        # Tighter constraints for higher priority
        # Set constraints based on priority
        # Tighter constraints for higher priority
        priority_multipliers = {
            "Critical": {"budget": 1.30, "time": 0.85, "delay": 0.85},
            "High": {"budget": 1.20, "time": 0.95, "delay": 0.95},
            "Medium": {"budget": 1.10, "time": 1.05, "delay": 1.05},
            "Low": {"budget": 1.05, "time": 1.15, "delay": 1.15},
        }
        pm = priority_multipliers[priority]

        # Number of alternative routes: 3-5
        num_routes = rng.integers(3, 6)

        # Generate routes first to set constraints relative to actual values
        route_costs = []
        route_times = []
        route_delays = []

        temp_routes = []
        for j in range(num_routes):
            route_id = f"{shipment_id}-R{j+1:02d}"

            # Generate varied routes
            # Route archetypes: cheap-slow, fast-expensive, balanced, risky-cheap, reliable
            archetype = j % 5
            
            if archetype == 0:  # Cheap but slow
                cost = rng.uniform(base_cost_range[0], base_cost_range[0] + (base_cost_range[1] - base_cost_range[0]) * 0.3)
                transit = rng.uniform(base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.6, base_time_range[1])
                delay = rng.uniform(base_delay_range[0] + 0.05, base_delay_range[1] * 0.8)
            elif archetype == 1:  # Fast but expensive
                cost = rng.uniform(base_cost_range[0] + (base_cost_range[1] - base_cost_range[0]) * 0.6, base_cost_range[1])
                transit = rng.uniform(base_time_range[0], base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.35)
                delay = rng.uniform(base_delay_range[0], base_delay_range[1] * 0.5)
            elif archetype == 2:  # Balanced
                cost = rng.uniform(base_cost_range[0] + (base_cost_range[1] - base_cost_range[0]) * 0.3,
                                   base_cost_range[0] + (base_cost_range[1] - base_cost_range[0]) * 0.7)
                transit = rng.uniform(base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.3,
                                      base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.7)
                delay = rng.uniform(base_delay_range[0] + 0.02, base_delay_range[1] * 0.6)
            elif archetype == 3:  # Risky but cheap
                cost = rng.uniform(base_cost_range[0], base_cost_range[0] + (base_cost_range[1] - base_cost_range[0]) * 0.25)
                transit = rng.uniform(base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.3,
                                      base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.7)
                delay = rng.uniform(base_delay_range[1] * 0.6, base_delay_range[1])
            else:  # archetype == 4: Reliable premium
                cost = rng.uniform(base_cost_range[0] + (base_cost_range[1] - base_cost_range[0]) * 0.5, base_cost_range[1])
                transit = rng.uniform(base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.2,
                                      base_time_range[0] + (base_time_range[1] - base_time_range[0]) * 0.5)
                delay = rng.uniform(base_delay_range[0], base_delay_range[0] + 0.05)

            cost = round(cost, 2)
            transit = round(transit, 1)
            delay = round(min(delay, 0.55), 3)

            # Congestion risk: correlated with delay but with noise
            congestion = round(np.clip(delay * rng.uniform(0.7, 1.5) + rng.uniform(-0.05, 0.1), 0.01, 0.95), 3)

            # Geopolitical risk: based on origin/destination regions
            geo_base = max(
                LOGISTICS_HUBS[origin]["geo_risk_base"],
                LOGISTICS_HUBS[destination]["geo_risk_base"],
            )
            geo_risk = round(np.clip(geo_base + rng.uniform(-0.08, 0.12), 0.01, 0.90), 3)

            # Intermediate hubs
            hubs = _generate_intermediate_hubs(origin, destination, rng)

            route_costs.append(cost)
            route_times.append(transit)
            route_delays.append(delay)

            temp_routes.append({
                "shipment_id": shipment_id,
                "route_id": route_id,
                "intermediate_hubs": hubs,
                "shipping_cost": cost,
                "transit_days": transit,
                "delay_probability": delay,
                "congestion_risk": congestion,
                "geopolitical_risk": geo_risk,
            })

        routes_records.extend(temp_routes)

        # Set shipment constraints relative to route characteristics (using 75th percentile as base reference)
        cost_ref = np.percentile(route_costs, 75)
        time_ref = np.percentile(route_times, 75)
        delay_ref = np.percentile(route_delays, 75)

        max_budget = round(cost_ref * pm["budget"], 2)
        max_transit = round(time_ref * pm["time"], 1)
        max_delay = round(min(delay_ref * pm["delay"], 0.55), 3)

        # For ~3-4 shipments, make constraints extremely tight (no feasible route)
        if i in [5, 18, 33]:
            max_budget = round(min(route_costs) * 0.5, 2)
            max_transit = round(min(route_times) * 0.5, 1)
            max_delay = round(min(route_delays) * 0.3, 3)

        shipments_records.append({
            "shipment_id": shipment_id,
            "origin": origin,
            "destination": destination,
            "priority": priority,
            "max_budget": max_budget,
            "max_transit_days": max_transit,
            "max_delay_probability": max_delay,
        })

    shipments_df = pd.DataFrame(shipments_records)
    routes_df = pd.DataFrame(routes_records)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    shipments_df.to_csv(os.path.join(output_dir, "shipments.csv"), index=False)
    routes_df.to_csv(os.path.join(output_dir, "routes.csv"), index=False)

    return shipments_df, routes_df


def load_dataset(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the shipment and route datasets from CSV.

    Parameters
    ----------
    data_dir : str
        Directory containing shipments.csv and routes.csv.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        shipments_df, routes_df
    """
    shipments_path = os.path.join(data_dir, "shipments.csv")
    routes_path = os.path.join(data_dir, "routes.csv")

    if not os.path.exists(shipments_path) or not os.path.exists(routes_path):
        raise FileNotFoundError(
            f"Dataset files not found in '{data_dir}'. "
            "Run data generation first."
        )

    shipments_df = pd.read_csv(shipments_path)
    routes_df = pd.read_csv(routes_path)

    return shipments_df, routes_df


if __name__ == "__main__":
    shipments, routes = generate_dataset()
    print(f"Generated {len(shipments)} shipments with {len(routes)} total routes.")
    print(f"\nShipments sample:\n{shipments.head()}")
    print(f"\nRoutes sample:\n{routes.head()}")
    print(f"\nRoutes per shipment:\n{routes.groupby('shipment_id').size().describe()}")
