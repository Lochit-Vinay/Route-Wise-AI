"""
Multi-Objective Optimization Engine for RouteWise AI.

Implements a weighted-sum multi-objective optimization approach
with min-max normalization for route scoring and ranking.

Methodology:
-----------
1. Normalize cost, transit time, and composite risk to [0, 1] using
   min-max normalization across available alternatives.
2. Compute a weighted composite score:
   Score = w_cost × C_norm + w_time × T_norm + w_risk × R_norm
3. Lower scores indicate more desirable routes.
4. Only feasible routes (satisfying all hard constraints) are considered.

Limitations:
-----------
- Weighted-sum optimization is a decision-support heuristic, not a
  guarantee of globally optimal logistics routing.
- The method assumes linear trade-offs between objectives.
- Normalization is relative to the available alternatives for each
  shipment, so scores are not comparable across shipments.
- Equal spacing on the weight scale does not imply equal importance
  in real-world impact.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class OptimizationWeights:
    """Configurable optimization weights."""
    cost: float = 0.40
    time: float = 0.35
    risk: float = 0.25

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Ensure weights are valid and sum to 1."""
        for name, val in [("cost", self.cost), ("time", self.time), ("risk", self.risk)]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"Weight '{name}' must be numeric, got {type(val).__name__}")
            if val < 0 or val > 1:
                raise ValueError(f"Weight '{name}' must be between 0 and 1, got {val}")

        total = round(self.cost + self.time + self.risk, 10)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f} "
                f"(cost={self.cost}, time={self.time}, risk={self.risk})"
            )

    def to_dict(self) -> Dict[str, float]:
        return {"cost": self.cost, "time": self.time, "risk": self.risk}


@dataclass
class NormalizationResult:
    """Result of min-max normalization for a set of routes."""
    cost_min: float
    cost_max: float
    time_min: float
    time_max: float
    risk_min: float
    risk_max: float
    normalized_values: pd.DataFrame


@dataclass
class RouteScore:
    """Complete scoring breakdown for a single route."""
    route_id: str
    raw_cost: float
    raw_time: float
    raw_risk: float
    norm_cost: float
    norm_time: float
    norm_risk: float
    weighted_cost: float
    weighted_time: float
    weighted_risk: float
    total_score: float
    rank: int = 0
    is_pareto_optimal: bool = True


def normalize_metrics(
    routes_df: pd.DataFrame,
    cost_col: str = "shipping_cost",
    time_col: str = "transit_days",
    risk_col: str = "composite_risk",
) -> NormalizationResult:
    """
    Apply min-max normalization to cost, time, and risk metrics.

    For each metric M:
        M_norm = (M - M_min) / (M_max - M_min)

    If M_max == M_min (all values identical), M_norm is set to 0.0
    to avoid division by zero.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route data with cost, time, and risk columns.
    cost_col, time_col, risk_col : str
        Column names for the three objectives.

    Returns
    -------
    NormalizationResult
        Contains normalization bounds and normalized DataFrame.
    """
    df = routes_df.copy()

    cost_min, cost_max = df[cost_col].min(), df[cost_col].max()
    time_min, time_max = df[time_col].min(), df[time_col].max()
    risk_min, risk_max = df[risk_col].min(), df[risk_col].max()

    # Min-max normalization with division-by-zero protection
    df["cost_normalized"] = _safe_normalize(df[cost_col], cost_min, cost_max)
    df["time_normalized"] = _safe_normalize(df[time_col], time_min, time_max)
    df["risk_normalized"] = _safe_normalize(df[risk_col], risk_min, risk_max)

    return NormalizationResult(
        cost_min=cost_min,
        cost_max=cost_max,
        time_min=time_min,
        time_max=time_max,
        risk_min=risk_min,
        risk_max=risk_max,
        normalized_values=df,
    )


def _safe_normalize(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """
    Safely normalize a series using min-max scaling.

    Returns 0.0 for all values if range is zero.
    """
    range_val = max_val - min_val
    if abs(range_val) < 1e-10:
        return pd.Series(0.0, index=series.index)
    return ((series - min_val) / range_val).round(6)


def compute_weighted_scores(
    routes_df: pd.DataFrame,
    weights: OptimizationWeights,
) -> pd.DataFrame:
    """
    Compute weighted optimization scores for routes.

    Score = w_cost × C_norm + w_time × T_norm + w_risk × R_norm

    Lower scores are better (more desirable).

    Parameters
    ----------
    routes_df : pd.DataFrame
        DataFrame with normalized columns (cost_normalized, etc.).
    weights : OptimizationWeights
        The objective weights.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional score columns.
    """
    df = routes_df.copy()

    df["weighted_cost"] = (weights.cost * df["cost_normalized"]).round(6)
    df["weighted_time"] = (weights.time * df["time_normalized"]).round(6)
    df["weighted_risk"] = (weights.risk * df["risk_normalized"]).round(6)

    df["optimization_score"] = (
        df["weighted_cost"] + df["weighted_time"] + df["weighted_risk"]
    ).round(6)

    # Rank routes (1 = best)
    df["rank"] = df["optimization_score"].rank(method="min").astype(int)
    df = df.sort_values("optimization_score").reset_index(drop=True)

    return df


def compute_pareto_frontier(
    routes_df: pd.DataFrame,
    cost_col: str = "shipping_cost",
    time_col: str = "transit_days",
    risk_col: str = "composite_risk",
) -> pd.Series:
    """
    Determine which routes in routes_df are Pareto-optimal (non-dominated).

    A route A dominates route B (A > B) if:
        cost(A) <= cost(B) and time(A) <= time(B) and risk(A) <= risk(B)
        with at least one strict inequality (<).

    Returns
    -------
    pd.Series
        Boolean Series indexed identically to routes_df, where True indicates
        the route is Pareto-optimal (not dominated by any other route).
    """
    if routes_df.empty:
        return pd.Series(dtype=bool)

    n = len(routes_df)
    if n == 1:
        return pd.Series([True], index=routes_df.index)

    costs = routes_df[cost_col].to_numpy()
    times = routes_df[time_col].to_numpy()

    if risk_col in routes_df.columns:
        risks = routes_df[risk_col].to_numpy()
    elif "delay_probability" in routes_df.columns:
        risks = routes_df["delay_probability"].to_numpy()
    else:
        risks = np.zeros(n)

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if route j strictly dominates route i
            j_dominates_i = (
                (costs[j] <= costs[i])
                and (times[j] <= times[i])
                and (risks[j] <= risks[i])
                and (
                    (costs[j] < costs[i])
                    or (times[j] < times[i])
                    or (risks[j] < risks[i])
                )
            )
            if j_dominates_i:
                is_pareto[i] = False
                break

    return pd.Series(is_pareto, index=routes_df.index)


def get_priority_weights(priority: str) -> OptimizationWeights:
    """
    Return tailored optimization weights based on shipment priority level.

    - Critical: Speed and risk mitigation are paramount.
    - High: Balanced focus on transit time and risk with moderate budget sensitivity.
    - Medium: Default balanced weighting across cost, time, and risk.
    - Low: Cost economy is primary; relaxed time and risk tolerances.
    """
    p = str(priority).strip().capitalize()
    if p == "Critical":
        return OptimizationWeights(cost=0.20, time=0.45, risk=0.35)
    elif p == "High":
        return OptimizationWeights(cost=0.30, time=0.40, risk=0.30)
    elif p == "Low":
        return OptimizationWeights(cost=0.60, time=0.25, risk=0.15)
    else:  # Medium or default
        return OptimizationWeights(cost=0.40, time=0.35, risk=0.25)


def optimize_routes(
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
    cost_col: str = "shipping_cost",
    time_col: str = "transit_days",
    risk_col: str = "composite_risk",
) -> Tuple[pd.DataFrame, NormalizationResult]:
    """
    Full optimization pipeline: normalize, score, and rank routes.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route alternatives (should already be filtered for feasibility
        if hard constraints are being enforced).
    weights : OptimizationWeights, optional
        Objective weights. Defaults to (0.40, 0.35, 0.25).
    cost_col, time_col, risk_col : str
        Column names for objectives.

    Returns
    -------
    Tuple[pd.DataFrame, NormalizationResult]
        Scored and ranked routes, and normalization metadata.
    """
    if weights is None:
        weights = OptimizationWeights()

    if routes_df.empty:
        empty_df = routes_df.copy()
        empty_df["is_pareto_optimal"] = pd.Series(dtype=bool)
        return empty_df, NormalizationResult(
            cost_min=0, cost_max=0, time_min=0, time_max=0,
            risk_min=0, risk_max=0, normalized_values=empty_df,
        )

    # Normalize
    norm_result = normalize_metrics(routes_df, cost_col, time_col, risk_col)

    # Score and rank
    scored_df = compute_weighted_scores(norm_result.normalized_values, weights)

    # Pareto-optimality identification
    scored_df["is_pareto_optimal"] = compute_pareto_frontier(
        scored_df, cost_col=cost_col, time_col=time_col, risk_col=risk_col
    )

    return scored_df, norm_result


def get_route_scores(
    routes_df: pd.DataFrame,
    weights: Optional[OptimizationWeights] = None,
) -> List[RouteScore]:
    """
    Get detailed score breakdowns for all routes.

    Returns
    -------
    List[RouteScore]
        Ordered list of route scores (best first).
    """
    scored_df, _ = optimize_routes(routes_df, weights)

    scores = []
    for _, row in scored_df.iterrows():
        scores.append(RouteScore(
            route_id=row["route_id"],
            raw_cost=row["shipping_cost"],
            raw_time=row["transit_days"],
            raw_risk=row.get("composite_risk", 0),
            norm_cost=row["cost_normalized"],
            norm_time=row["time_normalized"],
            norm_risk=row["risk_normalized"],
            weighted_cost=row["weighted_cost"],
            weighted_time=row["weighted_time"],
            weighted_risk=row["weighted_risk"],
            total_score=row["optimization_score"],
            rank=row["rank"],
            is_pareto_optimal=bool(row.get("is_pareto_optimal", True)),
        ))

    return scores
