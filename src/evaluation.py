"""
Evaluation Module for RouteWise AI.

Evaluates the multi-objective optimization engine across the entire
synthetic dataset and compares it with baseline strategies.

Baseline Strategies:
-------------------
1. Cheapest Route: Select feasible route with lowest shipping cost.
2. Fastest Route: Select feasible route with lowest transit time.
3. Lowest-Risk Route: Select feasible route with lowest delay probability.

The weighted optimization approach is a decision-support framework,
not a proof of globally optimal logistics routing. Baseline comparisons
demonstrate the trade-offs inherent in different selection strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.optimizer import OptimizationWeights, optimize_routes
from src.preprocessing import preprocess_routes
from src.constraints import get_feasible_routes, evaluate_shipment_constraints
from src.recommender import recommend_route, Recommendation


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single strategy."""
    strategy_name: str
    shipments_evaluated: int
    shipments_with_feasible: int
    shipments_no_feasible: int
    pct_constraints_satisfied: float
    avg_cost: float
    avg_transit_time: float
    avg_delay_probability: float
    avg_optimization_score: Optional[float]
    selected_routes: Dict[str, str]  # shipment_id -> route_id


def evaluate_baseline_cheapest(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> EvaluationResult:
    """Select the feasible route with the lowest cost for each shipment."""
    return _evaluate_baseline(shipments_df, routes_df, "shipping_cost", "Cheapest Route", weights)


def evaluate_baseline_fastest(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> EvaluationResult:
    """Select the feasible route with the lowest transit time for each shipment."""
    return _evaluate_baseline(shipments_df, routes_df, "transit_days", "Fastest Route", weights)


def evaluate_baseline_lowest_risk(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> EvaluationResult:
    """Select the feasible route with the lowest delay probability for each shipment."""
    return _evaluate_baseline(shipments_df, routes_df, "delay_probability", "Lowest-Risk Route", weights)


def evaluate_weighted_optimization(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> EvaluationResult:
    """
    Evaluate the weighted multi-objective optimization strategy.

    Parameters
    ----------
    shipments_df : pd.DataFrame
        All shipments.
    routes_df : pd.DataFrame
        All routes.
    weights : OptimizationWeights, optional
        Weights for optimization.

    Returns
    -------
    EvaluationResult
        Evaluation metrics for the weighted optimization strategy.
    """
    if weights is None:
        weights = OptimizationWeights()

    selected_routes = {}
    costs = []
    transits = []
    delays = []
    scores = []
    feasible_count = 0
    no_feasible_count = 0

    for _, shipment in shipments_df.iterrows():
        sid = shipment["shipment_id"]
        ship_routes = routes_df[routes_df["shipment_id"] == sid].copy()

        processed = preprocess_routes(ship_routes)
        feasible = get_feasible_routes(
            processed,
            shipment["max_budget"],
            shipment["max_transit_days"],
            shipment["max_delay_probability"],
        )

        if feasible.empty:
            no_feasible_count += 1
            selected_routes[sid] = None
            continue

        feasible_count += 1
        scored_df, _ = optimize_routes(feasible, weights)
        best = scored_df.iloc[0]

        selected_routes[sid] = best["route_id"]
        costs.append(best["shipping_cost"])
        transits.append(best["transit_days"])
        delays.append(best["delay_probability"])
        scores.append(best["optimization_score"])

    total = len(shipments_df)
    pct_satisfied = (feasible_count / total * 100) if total > 0 else 0

    return EvaluationResult(
        strategy_name="Weighted Optimization",
        shipments_evaluated=total,
        shipments_with_feasible=feasible_count,
        shipments_no_feasible=no_feasible_count,
        pct_constraints_satisfied=round(pct_satisfied, 1),
        avg_cost=round(np.mean(costs), 2) if costs else 0,
        avg_transit_time=round(np.mean(transits), 1) if transits else 0,
        avg_delay_probability=round(np.mean(delays), 4) if delays else 0,
        avg_optimization_score=round(np.mean(scores), 4) if scores else None,
        selected_routes=selected_routes,
    )


def _evaluate_baseline(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    sort_column: str,
    strategy_name: str,
    weights: Optional[OptimizationWeights] = None,
) -> EvaluationResult:
    """
    Generic baseline evaluation: select feasible route with lowest value
    in the specified column.
    """
    if weights is None:
        weights = OptimizationWeights()

    selected_routes = {}
    costs = []
    transits = []
    delays = []
    feasible_count = 0
    no_feasible_count = 0

    for _, shipment in shipments_df.iterrows():
        sid = shipment["shipment_id"]
        ship_routes = routes_df[routes_df["shipment_id"] == sid].copy()

        processed = preprocess_routes(ship_routes)
        feasible = get_feasible_routes(
            processed,
            shipment["max_budget"],
            shipment["max_transit_days"],
            shipment["max_delay_probability"],
        )

        if feasible.empty:
            no_feasible_count += 1
            selected_routes[sid] = None
            continue

        feasible_count += 1
        best = feasible.sort_values(sort_column).iloc[0]

        selected_routes[sid] = best["route_id"]
        costs.append(best["shipping_cost"])
        transits.append(best["transit_days"])
        delays.append(best["delay_probability"])

    total = len(shipments_df)
    pct_satisfied = (feasible_count / total * 100) if total > 0 else 0

    # Calculate optimization scores for baseline selections using provided weights
    opt_scores = []
    for _, shipment in shipments_df.iterrows():
        sid = shipment["shipment_id"]
        if selected_routes.get(sid) is None:
            continue
        ship_routes = routes_df[routes_df["shipment_id"] == sid].copy()
        processed = preprocess_routes(ship_routes)
        feasible = get_feasible_routes(
            processed,
            shipment["max_budget"],
            shipment["max_transit_days"],
            shipment["max_delay_probability"],
        )
        if feasible.empty:
            continue
        scored_df, _ = optimize_routes(feasible, weights)
        selected = scored_df[scored_df["route_id"] == selected_routes[sid]]
        if not selected.empty:
            opt_scores.append(selected.iloc[0]["optimization_score"])

    return EvaluationResult(
        strategy_name=strategy_name,
        shipments_evaluated=total,
        shipments_with_feasible=feasible_count,
        shipments_no_feasible=no_feasible_count,
        pct_constraints_satisfied=round(pct_satisfied, 1),
        avg_cost=round(np.mean(costs), 2) if costs else 0,
        avg_transit_time=round(np.mean(transits), 1) if transits else 0,
        avg_delay_probability=round(np.mean(delays), 4) if delays else 0,
        avg_optimization_score=round(np.mean(opt_scores), 4) if opt_scores else None,
        selected_routes=selected_routes,
    )


def run_full_evaluation(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> Dict[str, EvaluationResult]:
    """
    Run complete evaluation: weighted optimization + all baselines.

    Returns
    -------
    Dict[str, EvaluationResult]
        Results keyed by strategy name.
    """
    results = {}

    results["Weighted Optimization"] = evaluate_weighted_optimization(
        shipments_df, routes_df, weights
    )
    results["Cheapest Route"] = evaluate_baseline_cheapest(shipments_df, routes_df, weights)
    results["Fastest Route"] = evaluate_baseline_fastest(shipments_df, routes_df, weights)
    results["Lowest-Risk Route"] = evaluate_baseline_lowest_risk(shipments_df, routes_df, weights)

    return results


def evaluation_to_dataframe(results: Dict[str, EvaluationResult]) -> pd.DataFrame:
    """Convert evaluation results to a comparison DataFrame."""
    rows = []
    for name, result in results.items():
        rows.append({
            "Strategy": result.strategy_name,
            "Shipments Evaluated": result.shipments_evaluated,
            "Feasible": result.shipments_with_feasible,
            "No Feasible": result.shipments_no_feasible,
            "Constraints Satisfied (%)": result.pct_constraints_satisfied,
            "Avg Cost ($)": result.avg_cost,
            "Avg Transit (days)": result.avg_transit_time,
            "Avg Delay Prob": result.avg_delay_probability,
            "Avg Opt Score": result.avg_optimization_score,
        })

    return pd.DataFrame(rows)
