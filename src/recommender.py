"""
Recommendation Engine for RouteWise AI.

Orchestrates constraint checking, optimization scoring,
and route recommendation for a given shipment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.constraints import (
    evaluate_shipment_constraints,
    get_feasible_routes,
    find_closest_infeasible,
    ConstraintResult,
)
from src.optimizer import (
    optimize_routes,
    OptimizationWeights,
    NormalizationResult,
    RouteScore,
    get_route_scores,
)
from src.preprocessing import preprocess_routes


@dataclass
class Recommendation:
    """Complete recommendation result for a shipment."""
    shipment_id: str
    has_feasible_routes: bool
    recommended_route_id: Optional[str]
    recommended_route: Optional[pd.Series]
    all_routes_scored: pd.DataFrame
    constraint_results: List[ConstraintResult]
    feasible_count: int
    total_routes: int
    weights_used: OptimizationWeights
    normalization: Optional[NormalizationResult]
    explanation: str
    score_breakdown: Optional[Dict] = None
    closest_infeasible: Optional[pd.Series] = None
    is_pareto_optimal: bool = True


def recommend_route(
    shipment: pd.Series,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
    custom_budget: Optional[float] = None,
    custom_transit: Optional[float] = None,
    custom_delay: Optional[float] = None,
) -> Recommendation:
    """
    Generate a route recommendation for a shipment.

    Pipeline:
    1. Preprocess routes (compute composite risk).
    2. Evaluate hard constraints.
    3. Filter to feasible routes.
    4. Normalize and score feasible routes.
    5. Select the best (lowest score) route.
    6. Generate explanation.

    Parameters
    ----------
    shipment : pd.Series
        Shipment data including constraints.
    routes_df : pd.DataFrame
        All route alternatives for this shipment.
    weights : OptimizationWeights, optional
        User-defined weights.
    custom_budget, custom_transit, custom_delay : float, optional
        Override shipment constraints.

    Returns
    -------
    Recommendation
        Complete recommendation with explanation.
    """
    if weights is None:
        weights = OptimizationWeights()

    shipment_id = shipment["shipment_id"]

    # Use custom or default constraints
    max_budget = custom_budget if custom_budget is not None else shipment["max_budget"]
    max_transit = custom_transit if custom_transit is not None else shipment["max_transit_days"]
    max_delay = custom_delay if custom_delay is not None else shipment["max_delay_probability"]

    # Step 1: Preprocess
    processed_routes = preprocess_routes(routes_df)

    # Step 2: Evaluate constraints
    routes_with_feasibility, constraint_results = evaluate_shipment_constraints(
        processed_routes, max_budget, max_transit, max_delay
    )

    feasible = routes_with_feasibility[routes_with_feasibility["is_feasible"]].copy()
    feasible_count = len(feasible)
    total_routes = len(routes_with_feasibility)

    # Step 3: Handle no feasible routes
    if feasible_count == 0:
        closest = find_closest_infeasible(
            processed_routes, max_budget, max_transit, max_delay
        )

        explanation = _generate_no_feasible_explanation(
            shipment_id, constraint_results, closest,
            max_budget, max_transit, max_delay,
        )

        return Recommendation(
            shipment_id=shipment_id,
            has_feasible_routes=False,
            recommended_route_id=None,
            recommended_route=None,
            all_routes_scored=routes_with_feasibility,
            constraint_results=constraint_results,
            feasible_count=0,
            total_routes=total_routes,
            weights_used=weights,
            normalization=None,
            explanation=explanation,
            closest_infeasible=closest,
            is_pareto_optimal=False,
        )

    # Step 4: Optimize feasible routes
    scored_df, norm_result = optimize_routes(feasible, weights)

    # Step 5: Select best route
    best = scored_df.iloc[0]
    best_route_id = best["route_id"]

    # Merge scores back into full route set for display
    all_routes = routes_with_feasibility.copy()
    score_cols = ["cost_normalized", "time_normalized", "risk_normalized",
                  "weighted_cost", "weighted_time", "weighted_risk",
                  "optimization_score", "rank", "is_pareto_optimal"]
    for col in score_cols:
        if col in scored_df.columns:
            score_map = scored_df.set_index("route_id")[col]
            all_routes[col] = all_routes["route_id"].map(score_map)

    # Score breakdown for the best route
    score_breakdown = {
        "cost_normalized": float(best["cost_normalized"]),
        "time_normalized": float(best["time_normalized"]),
        "risk_normalized": float(best["risk_normalized"]),
        "weighted_cost": float(best["weighted_cost"]),
        "weighted_time": float(best["weighted_time"]),
        "weighted_risk": float(best["weighted_risk"]),
        "total_score": float(best["optimization_score"]),
        "is_pareto_optimal": bool(best.get("is_pareto_optimal", True)),
    }

    # Step 6: Generate explanation
    runner_up = scored_df.iloc[1] if len(scored_df) > 1 else None
    explanation = _generate_recommendation_explanation(
        best, runner_up, weights, norm_result, feasible_count, total_routes, constraint_results
    )

    return Recommendation(
        shipment_id=shipment_id,
        has_feasible_routes=True,
        recommended_route_id=best_route_id,
        recommended_route=best,
        all_routes_scored=all_routes,
        constraint_results=constraint_results,
        feasible_count=feasible_count,
        total_routes=total_routes,
        weights_used=weights,
        normalization=norm_result,
        explanation=explanation,
        score_breakdown=score_breakdown,
        is_pareto_optimal=bool(best.get("is_pareto_optimal", True)),
    )


def recommend_all_shipments(
    shipments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> List[Recommendation]:
    """
    Generate recommendations for all shipments.

    Parameters
    ----------
    shipments_df : pd.DataFrame
        All shipments.
    routes_df : pd.DataFrame
        All routes.
    weights : OptimizationWeights, optional
        Weights to use for all shipments.

    Returns
    -------
    List[Recommendation]
        List of recommendations, one per shipment.
    """
    recommendations = []
    for _, shipment in shipments_df.iterrows():
        ship_routes = routes_df[routes_df["shipment_id"] == shipment["shipment_id"]].copy()
        rec = recommend_route(shipment, ship_routes, weights)
        recommendations.append(rec)
    return recommendations


def _generate_recommendation_explanation(
    best: pd.Series,
    runner_up: Optional[pd.Series],
    weights: OptimizationWeights,
    norm_result: NormalizationResult,
    feasible_count: int,
    total_routes: int,
    constraint_results: Optional[List[ConstraintResult]] = None,
) -> str:
    """Generate a plain-English explanation for the recommendation."""
    parts = []

    parts.append(
        f"**{best['route_id']}** is recommended as the optimal route based on "
        f"the weighted multi-objective evaluation."
    )

    parts.append(
        f"\nOut of {total_routes} available routes, {feasible_count} satisfied all "
        f"hard constraints (budget, transit time, and delay probability limits)."
    )

    # Score breakdown
    parts.append(
        f"\n**Score Breakdown** (lower is better):"
    )
    parts.append(
        f"- Cost component: {best['cost_normalized']:.4f} normalized "
        f"× {weights.cost:.2f} weight = **{best['weighted_cost']:.4f}**"
    )
    parts.append(
        f"- Time component: {best['time_normalized']:.4f} normalized "
        f"× {weights.time:.2f} weight = **{best['weighted_time']:.4f}**"
    )
    parts.append(
        f"- Risk component: {best['risk_normalized']:.4f} normalized "
        f"× {weights.risk:.2f} weight = **{best['weighted_risk']:.4f}**"
    )
    parts.append(
        f"- **Total optimization score: {best['optimization_score']:.4f}**"
    )

    # Raw values & Pareto efficiency
    parts.append(
        f"\n**Route Details:**"
    )
    parts.append(f"- Shipping cost: ${best['shipping_cost']:,.2f}")
    parts.append(f"- Transit time: {best['transit_days']:.1f} days")
    parts.append(f"- Delay probability: {best['delay_probability']:.1%}")
    if "composite_risk" in best.index:
        parts.append(f"- Composite risk: {best['composite_risk']:.4f}")

    if best.get("is_pareto_optimal", True):
        parts.append("- **Pareto Efficiency:** ⭐ Pareto-optimal (non-dominated across cost, time, and risk).")

    # Soft constraint / capacity warnings
    if constraint_results:
        for cr in constraint_results:
            if cr.route_id == best["route_id"] and getattr(cr, "soft_warnings", None):
                parts.append(f"\n⚠️ **Capacity / Margin Advisory:** {'; '.join(cr.soft_warnings)}")

    # Comparison with runner-up
    if runner_up is not None:
        diff = runner_up["optimization_score"] - best["optimization_score"]
        parts.append(
            f"\n**Comparison with next-best alternative ({runner_up['route_id']}):**"
        )
        parts.append(
            f"- {runner_up['route_id']} scored {runner_up['optimization_score']:.4f} "
            f"(+{diff:.4f} higher than recommended route)."
        )

        cost_diff = runner_up["shipping_cost"] - best["shipping_cost"]
        time_diff = runner_up["transit_days"] - best["transit_days"]
        delay_diff = runner_up["delay_probability"] - best["delay_probability"]

        metrics_comp = []
        if cost_diff != 0:
            metrics_comp.append(f"{'+' if cost_diff > 0 else ''}${cost_diff:,.2f} cost")
        if time_diff != 0:
            metrics_comp.append(f"{'+' if time_diff > 0 else ''}{time_diff:.1f} d transit")
        if delay_diff != 0:
            metrics_comp.append(f"{'+' if delay_diff > 0 else ''}{delay_diff:.1%} delay")

        if metrics_comp:
            parts.append(f"- Raw metrics variance: {', '.join(metrics_comp)}.")

        # Explain why the runner-up ranked lower
        reasons = []
        if runner_up["weighted_cost"] > best["weighted_cost"] + 0.005:
            reasons.append(f"higher cost contribution ({runner_up['weighted_cost']:.4f} vs {best['weighted_cost']:.4f})")
        if runner_up["weighted_time"] > best["weighted_time"] + 0.005:
            reasons.append(f"longer transit contribution ({runner_up['weighted_time']:.4f} vs {best['weighted_time']:.4f})")
        if runner_up["weighted_risk"] > best["weighted_risk"] + 0.005:
            reasons.append(f"higher risk contribution ({runner_up['weighted_risk']:.4f} vs {best['weighted_risk']:.4f})")

        if reasons:
            parts.append(f"- Ranked lower due to: {'; '.join(reasons)}.")
        else:
            parts.append(f"- The difference is marginal across all objectives.")

    # Methodology note
    parts.append(
        f"\n*Note: RouteWise AI uses a normalized multi-attribute utility framework. "
        f"Scores are relative to candidate alternatives for this shipment. "
        f"Adjust weights in the sidebar to simulate different organizational priorities.*"
    )

    return "\n".join(parts)


def _generate_no_feasible_explanation(
    shipment_id: str,
    constraint_results: List[ConstraintResult],
    closest: Optional[pd.Series],
    max_budget: float,
    max_transit: float,
    max_delay: float,
) -> str:
    """Generate explanation when no feasible route exists."""
    parts = []

    parts.append(
        f"⚠️ **No feasible route found for shipment {shipment_id}.**"
    )

    parts.append(
        f"\nAll {len(constraint_results)} route alternatives violate at least one "
        f"hard constraint:"
    )
    parts.append(f"- Maximum budget: ${max_budget:,.2f}")
    parts.append(f"- Maximum transit time: {max_transit:.1f} days")
    parts.append(f"- Maximum delay probability: {max_delay:.1%}")

    # Summarize violations
    parts.append(f"\n**Violation Summary:**")
    for result in constraint_results:
        if not result.is_feasible:
            violations_str = "; ".join(result.violated_constraints)
            parts.append(f"- {result.route_id}: {violations_str}")

    # Closest alternative and actionable relaxations
    if closest is not None:
        from src.constraints import compute_required_relaxations
        relaxations = compute_required_relaxations(closest, max_budget, max_transit, max_delay)

        parts.append(f"\n**Closest Alternative:** {closest['route_id']}")
        parts.append(f"- Cost: ${closest['shipping_cost']:,.2f} (limit: ${max_budget:,.2f})")
        parts.append(f"- Transit: {closest['transit_days']:.1f} days (limit: {max_transit:.1f} days)")
        parts.append(f"- Delay: {closest['delay_probability']:.1%} (limit: {max_delay:.1%})")

        recs = []
        if relaxations["budget"]["is_violated"]:
            recs.append(f"increase budget by ${relaxations['budget']['required_increase']:,.2f} (+{relaxations['budget']['pct_increase']}%)")
        if relaxations["transit"]["is_violated"]:
            recs.append(f"extend transit deadline by {relaxations['transit']['required_increase']:.1f} days (+{relaxations['transit']['pct_increase']}%)")
        if relaxations["delay"]["is_violated"]:
            recs.append(f"relax delay tolerance by {relaxations['delay']['required_increase']:.3f} (+{relaxations['delay']['pct_increase']}%)")

        if recs:
            parts.append(f"\n💡 **Actionable Concessions to Restore Feasibility for {closest['route_id']}:**")
            for r in recs:
                parts.append(f"- {r.capitalize()}")

    return "\n".join(parts)
