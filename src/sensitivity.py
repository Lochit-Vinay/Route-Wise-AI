"""
Sensitivity Analysis Module for RouteWise AI.

Analyzes how changes in optimization weights and constraints
affect route recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.optimizer import OptimizationWeights, optimize_routes
from src.preprocessing import preprocess_routes
from src.constraints import get_feasible_routes


def weight_sensitivity_analysis(
    routes_df: pd.DataFrame,
    max_budget: float,
    max_transit_days: float,
    max_delay_probability: float,
    weight_steps: int = 11,
) -> pd.DataFrame:
    """
    Perform weight sensitivity analysis.

    Tests different weight combinations and records which route
    is recommended for each.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route alternatives for a single shipment.
    max_budget, max_transit_days, max_delay_probability : float
        Hard constraints.
    weight_steps : int
        Number of steps per weight dimension (default 11 → 0.0 to 1.0 in 0.1).

    Returns
    -------
    pd.DataFrame
        Results with columns: cost_weight, time_weight, risk_weight,
        recommended_route, optimization_score.
    """
    processed = preprocess_routes(routes_df)
    feasible = get_feasible_routes(
        processed, max_budget, max_transit_days, max_delay_probability
    )

    if feasible.empty:
        return pd.DataFrame(columns=[
            "cost_weight", "time_weight", "risk_weight",
            "recommended_route", "optimization_score",
        ])

    results = []
    step_size = 1.0 / (weight_steps - 1) if weight_steps > 1 else 1.0

    for i in range(weight_steps):
        w_cost = round(i * step_size, 2)
        for j in range(weight_steps - i):
            w_time = round(j * step_size, 2)
            w_risk = round(1.0 - w_cost - w_time, 2)

            if w_risk < -1e-6 or w_risk > 1.0 + 1e-6:
                continue
            w_risk = max(0, min(1.0, w_risk))

            if abs(w_cost + w_time + w_risk - 1.0) > 1e-6:
                continue

            try:
                weights = OptimizationWeights(cost=w_cost, time=w_time, risk=w_risk)
                scored_df, _ = optimize_routes(feasible, weights)

                if not scored_df.empty:
                    best = scored_df.iloc[0]
                    results.append({
                        "cost_weight": w_cost,
                        "time_weight": w_time,
                        "risk_weight": w_risk,
                        "recommended_route": best["route_id"],
                        "optimization_score": best["optimization_score"],
                    })
            except (ValueError, TypeError):
                continue

    return pd.DataFrame(results)


def constraint_sensitivity(
    routes_df: pd.DataFrame,
    base_budget: float,
    base_transit: float,
    base_delay: float,
    weights: Optional[OptimizationWeights] = None,
    budget_range: Tuple[float, float] = (0.5, 2.0),
    transit_range: Tuple[float, float] = (0.5, 2.0),
    delay_range: Tuple[float, float] = (0.5, 2.0),
    steps: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze how changing constraint bounds affects recommendations.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route alternatives.
    base_budget, base_transit, base_delay : float
        Base constraint values.
    weights : OptimizationWeights, optional
        Weights for scoring.
    budget_range, transit_range, delay_range : Tuple[float, float]
        Multiplier ranges for each constraint.
    steps : int
        Number of steps within each range.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Sensitivity results for each constraint dimension.
    """
    if weights is None:
        weights = OptimizationWeights()

    processed = preprocess_routes(routes_df)
    results = {}

    # Budget sensitivity
    budget_results = []
    for mult in np.linspace(budget_range[0], budget_range[1], steps):
        test_budget = base_budget * mult
        feasible = get_feasible_routes(processed, test_budget, base_transit, base_delay)
        rec_route = None
        score = None
        if not feasible.empty:
            scored, _ = optimize_routes(feasible, weights)
            rec_route = scored.iloc[0]["route_id"]
            score = scored.iloc[0]["optimization_score"]
        budget_results.append({
            "multiplier": round(mult, 2),
            "budget_value": round(test_budget, 2),
            "feasible_count": len(feasible),
            "recommended_route": rec_route,
            "optimization_score": score,
        })
    results["budget"] = pd.DataFrame(budget_results)

    # Transit time sensitivity
    transit_results = []
    for mult in np.linspace(transit_range[0], transit_range[1], steps):
        test_transit = base_transit * mult
        feasible = get_feasible_routes(processed, base_budget, test_transit, base_delay)
        rec_route = None
        score = None
        if not feasible.empty:
            scored, _ = optimize_routes(feasible, weights)
            rec_route = scored.iloc[0]["route_id"]
            score = scored.iloc[0]["optimization_score"]
        transit_results.append({
            "multiplier": round(mult, 2),
            "transit_value": round(test_transit, 1),
            "feasible_count": len(feasible),
            "recommended_route": rec_route,
            "optimization_score": score,
        })
    results["transit"] = pd.DataFrame(transit_results)

    # Delay probability sensitivity
    delay_results = []
    for mult in np.linspace(delay_range[0], delay_range[1], steps):
        test_delay = min(base_delay * mult, 1.0)
        feasible = get_feasible_routes(processed, base_budget, base_transit, test_delay)
        rec_route = None
        score = None
        if not feasible.empty:
            scored, _ = optimize_routes(feasible, weights)
            rec_route = scored.iloc[0]["route_id"]
            score = scored.iloc[0]["optimization_score"]
        delay_results.append({
            "multiplier": round(mult, 2),
            "delay_value": round(test_delay, 3),
            "feasible_count": len(feasible),
            "recommended_route": rec_route,
            "optimization_score": score,
        })
    results["delay"] = pd.DataFrame(delay_results)

    return results


def summarize_weight_sensitivity(sensitivity_df: pd.DataFrame) -> Dict:
    """
    Summarize weight sensitivity results.

    Returns
    -------
    Dict
        Summary statistics including route frequency and stability.
    """
    if sensitivity_df.empty:
        return {"unique_routes": 0, "most_frequent": None, "stability": 0}

    route_counts = sensitivity_df["recommended_route"].value_counts()
    total = len(sensitivity_df)

    return {
        "unique_routes": len(route_counts),
        "most_frequent": route_counts.index[0],
        "most_frequent_pct": round(route_counts.iloc[0] / total * 100, 1),
        "route_distribution": route_counts.to_dict(),
        "stability": round(route_counts.iloc[0] / total, 3),
    }
