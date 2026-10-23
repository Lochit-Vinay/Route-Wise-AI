"""
Constraint Engine for RouteWise AI.

Handles hard constraint evaluation for shipment routes.
A route violating any hard constraint is marked infeasible
and excluded from the optimization recommendation.

Hard Constraints:
-----------------
1. Maximum shipping budget: Route cost must not exceed max_budget.
2. Maximum transit time: Route transit days must not exceed max_transit_days.
3. Maximum delay probability: Route delay probability must not exceed max_delay_probability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ConstraintResult:
    """Result of constraint evaluation for a single route."""
    route_id: str
    is_feasible: bool
    violated_constraints: List[str]
    constraint_details: Dict[str, Dict[str, float]]
    soft_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "route_id": self.route_id,
            "is_feasible": self.is_feasible,
            "violated_constraints": self.violated_constraints,
            "constraint_details": self.constraint_details,
            "soft_warnings": self.soft_warnings,
        }


def evaluate_route_constraints(
    route: pd.Series,
    max_budget: float,
    max_transit_days: float,
    max_delay_probability: float,
) -> ConstraintResult:
    """
    Evaluate hard constraints and soft margin thresholds for a single route.

    Parameters
    ----------
    route : pd.Series
        Route data with shipping_cost, transit_days, delay_probability.
    max_budget : float
        Maximum acceptable shipping cost.
    max_transit_days : float
        Maximum acceptable transit time in days.
    max_delay_probability : float
        Maximum acceptable delay probability (0-1).

    Returns
    -------
    ConstraintResult
        Result containing feasibility status, violation details, and soft warnings.
    """
    violated = []
    soft_warnings = []
    details = {}

    # Budget constraint
    cost = route["shipping_cost"]
    budget_ok = cost <= max_budget
    budget_margin = max_budget - cost
    details["budget"] = {
        "actual": cost,
        "limit": max_budget,
        "satisfied": budget_ok,
        "margin": budget_margin,
    }
    if not budget_ok:
        violated.append(f"Cost ${cost:,.2f} exceeds budget ${max_budget:,.2f}")
    elif max_budget > 0 and (budget_margin / max_budget) < 0.05:
        soft_warnings.append(f"Budget slack is tight (${budget_margin:,.2f} / {budget_margin/max_budget:.1%} remaining)")

    # Transit time constraint
    transit = route["transit_days"]
    transit_ok = transit <= max_transit_days
    transit_margin = max_transit_days - transit
    details["transit_time"] = {
        "actual": transit,
        "limit": max_transit_days,
        "satisfied": transit_ok,
        "margin": transit_margin,
    }
    if not transit_ok:
        violated.append(f"Transit {transit:.1f} days exceeds limit {max_transit_days:.1f} days")
    elif max_transit_days > 0 and (transit_margin / max_transit_days) < 0.05:
        soft_warnings.append(f"Transit slack is tight ({transit_margin:.1f} d / {transit_margin/max_transit_days:.1%} remaining)")

    # Delay probability constraint
    delay = route["delay_probability"]
    delay_ok = delay <= max_delay_probability
    delay_margin = max_delay_probability - delay
    details["delay_probability"] = {
        "actual": delay,
        "limit": max_delay_probability,
        "satisfied": delay_ok,
        "margin": delay_margin,
    }
    if not delay_ok:
        violated.append(f"Delay prob {delay:.3f} exceeds limit {max_delay_probability:.3f}")
    elif max_delay_probability > 0 and (delay_margin / max_delay_probability) < 0.05:
        soft_warnings.append(f"Delay margin is tight ({delay_margin:.3f} remaining)")

    is_feasible = len(violated) == 0

    return ConstraintResult(
        route_id=route["route_id"],
        is_feasible=is_feasible,
        violated_constraints=violated,
        constraint_details=details,
        soft_warnings=soft_warnings,
    )


def evaluate_shipment_constraints(
    routes_df: pd.DataFrame,
    max_budget: float,
    max_transit_days: float,
    max_delay_probability: float,
) -> Tuple[pd.DataFrame, List[ConstraintResult]]:
    """
    Evaluate constraints for all routes of a shipment.

    Parameters
    ----------
    routes_df : pd.DataFrame
        All route alternatives for a single shipment.
    max_budget : float
        Maximum budget constraint.
    max_transit_days : float
        Maximum transit time constraint.
    max_delay_probability : float
        Maximum delay probability constraint.

    Returns
    -------
    Tuple[pd.DataFrame, List[ConstraintResult]]
        DataFrame with feasibility column added, and list of constraint results.
    """
    results = []
    feasibility_flags = []

    for _, route in routes_df.iterrows():
        result = evaluate_route_constraints(
            route, max_budget, max_transit_days, max_delay_probability
        )
        results.append(result)
        feasibility_flags.append(result.is_feasible)

    routes_with_feasibility = routes_df.copy()
    routes_with_feasibility["is_feasible"] = feasibility_flags

    return routes_with_feasibility, results


def get_feasible_routes(
    routes_df: pd.DataFrame,
    max_budget: float,
    max_transit_days: float,
    max_delay_probability: float,
) -> pd.DataFrame:
    """
    Return only feasible routes that satisfy all hard constraints.

    Parameters
    ----------
    routes_df : pd.DataFrame
        All route alternatives.
    max_budget, max_transit_days, max_delay_probability : float
        Constraint thresholds.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only feasible routes.
    """
    routes_eval, _ = evaluate_shipment_constraints(
        routes_df, max_budget, max_transit_days, max_delay_probability
    )
    return routes_eval[routes_eval["is_feasible"]].copy()


def find_closest_infeasible(
    routes_df: pd.DataFrame,
    max_budget: float,
    max_transit_days: float,
    max_delay_probability: float,
) -> Optional[pd.Series]:
    """
    Find the infeasible route closest to satisfying constraints.

    Uses a normalized violation distance to rank infeasible routes.

    Returns
    -------
    Optional[pd.Series]
        The closest infeasible route, or None if all routes are feasible.
    """
    routes_eval, results = evaluate_shipment_constraints(
        routes_df, max_budget, max_transit_days, max_delay_probability
    )

    infeasible = routes_eval[~routes_eval["is_feasible"]].copy()
    if infeasible.empty:
        return None

    # Compute normalized violation distance
    violations = []
    for _, route in infeasible.iterrows():
        budget_violation = max(0, (route["shipping_cost"] - max_budget) / max_budget) if max_budget > 0 else 0
        time_violation = max(0, (route["transit_days"] - max_transit_days) / max_transit_days) if max_transit_days > 0 else 0
        delay_violation = max(0, (route["delay_probability"] - max_delay_probability) / max_delay_probability) if max_delay_probability > 0 else 0
        total_violation = budget_violation + time_violation + delay_violation
        violations.append(total_violation)

    infeasible = infeasible.copy()
    infeasible["violation_distance"] = violations

    closest = infeasible.loc[infeasible["violation_distance"].idxmin()]
    return closest


def compute_required_relaxations(
    target: Any,
    max_budget: Optional[float] = None,
    max_transit_days: Optional[float] = None,
    max_delay_probability: Optional[float] = None,
) -> Any:
    """
    Calculate the exact concessions needed to make an infeasible route feasible.

    Supports two calling conventions:
    1. compute_required_relaxations(route: pd.Series, max_budget, max_transit, max_delay)
       Returns Dict[str, Dict[str, float]]
    2. compute_required_relaxations(constraint_results: List[ConstraintResult])
       Returns List[Dict[str, Any]]
    """
    if isinstance(target, list):
        results = target
        if not results or any(r.is_feasible for r in results):
            return []

        # Find closest infeasible result
        infeasible = [r for r in results if not r.is_feasible]
        def calc_violation(res: ConstraintResult) -> float:
            tot = 0.0
            for d in res.constraint_details.values():
                if not d["satisfied"]:
                    limit = d["limit"]
                    actual = d["actual"]
                    tot += (actual - limit) / limit if limit > 0 else 1.0
            return tot

        closest_res = min(infeasible, key=calc_violation)
        relaxations = []
        name_map = {
            "budget": "Budget",
            "transit_time": "Transit Days",
            "delay_probability": "Delay Probability",
        }
        for key, name in name_map.items():
            if key in closest_res.constraint_details:
                d = closest_res.constraint_details[key]
                if not d["satisfied"]:
                    actual = float(d["actual"])
                    limit = float(d["limit"])
                    deficit = actual - limit
                    deficit_pct = round((deficit / limit * 100), 1) if limit > 0 else 0.0
                    relaxations.append({
                        "route_id": closest_res.route_id,
                        "metric": name,
                        "actual_value": actual,
                        "limit": limit,
                        "deficit": round(deficit, 2) if key == "budget" else round(deficit, 3),
                        "deficit_pct": deficit_pct,
                    })
        return relaxations

    # pd.Series route
    route = target
    cost_diff = max(0.0, float(route["shipping_cost"]) - (max_budget or 0.0))
    time_diff = max(0.0, float(route["transit_days"]) - (max_transit_days or 0.0))
    delay_diff = max(0.0, float(route["delay_probability"]) - (max_delay_probability or 0.0))

    return {
        "budget": {
            "required_increase": round(cost_diff, 2),
            "pct_increase": round((cost_diff / max_budget * 100), 1) if (max_budget and max_budget > 0) else 0.0,
            "is_violated": cost_diff > 0,
        },
        "transit": {
            "required_increase": round(time_diff, 1),
            "pct_increase": round((time_diff / max_transit_days * 100), 1) if (max_transit_days and max_transit_days > 0) else 0.0,
            "is_violated": time_diff > 0,
        },
        "delay": {
            "required_increase": round(delay_diff, 3),
            "pct_increase": round((delay_diff / max_delay_probability * 100), 1) if (max_delay_probability and max_delay_probability > 0) else 0.0,
            "is_violated": delay_diff > 0,
        },
    }
