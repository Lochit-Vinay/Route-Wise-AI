"""
Explainability Module for RouteWise AI.

Provides structured explanations and comparison analytics
for route recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.optimizer import OptimizationWeights


def generate_score_comparison_table(
    scored_routes: pd.DataFrame,
    weights: OptimizationWeights,
) -> pd.DataFrame:
    """
    Generate a formatted comparison table for all scored routes.

    Parameters
    ----------
    scored_routes : pd.DataFrame
        Routes with normalization and scoring columns.
    weights : OptimizationWeights
        Weights used for scoring.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    if scored_routes.empty:
        return pd.DataFrame()

    cols_to_display = {
        "route_id": "Route ID",
        "shipping_cost": "Cost ($)",
        "transit_days": "Transit (days)",
        "delay_probability": "Delay Prob",
        "composite_risk": "Composite Risk",
        "cost_normalized": "Cost (norm)",
        "time_normalized": "Time (norm)",
        "risk_normalized": "Risk (norm)",
        "optimization_score": "Score",
        "rank": "Rank",
        "is_pareto_optimal": "Pareto Optimal",
    }

    available_cols = [c for c in cols_to_display.keys() if c in scored_routes.columns]
    table = scored_routes[available_cols].copy()

    if "is_pareto_optimal" in table.columns:
        table["is_pareto_optimal"] = table["is_pareto_optimal"].apply(
            lambda x: "⭐ Yes" if x else "Dominated"
        )

    table = table.rename(columns={c: cols_to_display[c] for c in available_cols})
    return table


def explain_weight_impact(
    route: pd.Series,
    weights: OptimizationWeights,
) -> Dict[str, float]:
    """
    Calculate the percentage contribution of each objective
    to the total optimization score.

    Parameters
    ----------
    route : pd.Series
        A scored route.
    weights : OptimizationWeights
        The weights used.

    Returns
    -------
    Dict[str, float]
        Percentage contribution of each objective.
    """
    total = route.get("optimization_score", 0)
    if total == 0:
        return {"cost_pct": 33.3, "time_pct": 33.3, "risk_pct": 33.3}

    return {
        "cost_pct": round(route.get("weighted_cost", 0) / total * 100, 1),
        "time_pct": round(route.get("weighted_time", 0) / total * 100, 1),
        "risk_pct": round(route.get("weighted_risk", 0) / total * 100, 1),
    }


def generate_trade_off_summary(
    scored_routes: pd.DataFrame,
    recommended_route_id: str,
) -> str:
    """
    Generate a comprehensive trade-off summary comparing the recommended route
    with alternatives.
    """
    if scored_routes.empty or recommended_route_id is None:
        return "No routes available for trade-off analysis."

    recommended = scored_routes[scored_routes["route_id"] == recommended_route_id]
    if recommended.empty:
        return "Recommended route not found in scored routes."

    rec = recommended.iloc[0]
    others = scored_routes[scored_routes["route_id"] != recommended_route_id]

    parts = [f"### Trade-off Analysis for {recommended_route_id}\n"]

    if rec.get("is_pareto_optimal", True):
        parts.append(
            "- **Pareto Efficiency:** ⭐ This route is non-dominated. No alternative "
            "offers lower cost, faster transit, and lower risk simultaneously."
        )

    # Find cheapest, fastest, lowest-risk alternatives
    if not others.empty:
        cheapest = others.loc[others["shipping_cost"].idxmin()]
        if cheapest["shipping_cost"] < rec["shipping_cost"]:
            cost_diff = rec["shipping_cost"] - cheapest["shipping_cost"]
            time_diff = cheapest["transit_days"] - rec["transit_days"]
            parts.append(
                f"- **Cost vs Speed Trade-off:** {cheapest['route_id']} saves "
                f"${cost_diff:,.2f} on freight, but adds {time_diff:+.1f} transit days "
                f"and scores {cheapest.get('optimization_score', 'N/A'):.4f} overall "
                f"vs {rec.get('optimization_score', 'N/A'):.4f}."
            )

        fastest = others.loc[others["transit_days"].idxmin()]
        if fastest["transit_days"] < rec["transit_days"]:
            time_diff = rec["transit_days"] - fastest["transit_days"]
            cost_diff = fastest["shipping_cost"] - rec["shipping_cost"]
            parts.append(
                f"- **Speed vs Cost Trade-off:** {fastest['route_id']} is "
                f"{time_diff:.1f} days faster, but costs ${cost_diff:+,.2f} more "
                f"(overall score: {fastest.get('optimization_score', 'N/A'):.4f})."
            )

        if "composite_risk" in others.columns:
            safest = others.loc[others["composite_risk"].idxmin()]
            if safest["composite_risk"] < rec.get("composite_risk", 1):
                risk_diff = rec.get("composite_risk", 0) - safest["composite_risk"]
                parts.append(
                    f"- **Risk Trade-off:** {safest['route_id']} has "
                    f"{risk_diff:.4f} lower composite risk, but scores "
                    f"{safest.get('optimization_score', 'N/A'):.4f} overall."
                )

    if len(parts) == 1:
        parts.append("No significant trade-offs detected with alternatives.")

    return "\n".join(parts)


def format_constraint_status(
    constraint_results: list,
) -> pd.DataFrame:
    """
    Format constraint evaluation results into a display table including soft margin warnings.
    """
    rows = []
    for result in constraint_results:
        row = {"Route ID": result.route_id, "Feasible": "✅ Yes" if result.is_feasible else "❌ No"}
        for constraint_name, details in result.constraint_details.items():
            status = "✅" if details["satisfied"] else "❌"
            if constraint_name == "budget":
                row["Budget"] = f"{status} ${details['actual']:,.2f} / ${details['limit']:,.2f}"
            elif constraint_name == "transit_time":
                row["Transit"] = f"{status} {details['actual']:.1f} / {details['limit']:.1f} d"
            elif constraint_name == "delay_probability":
                row["Delay"] = f"{status} {details['actual']:.1%} / {details['limit']:.1%}"
        
        # Soft advisory column
        soft = getattr(result, "soft_warnings", [])
        row["Advisories"] = "⚠️ " + "; ".join(soft) if soft else "Optimal Margin"
        rows.append(row)

    return pd.DataFrame(rows)
