# RouteWise AI — Intelligent Multi-Objective Shipment Route Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-88%20Passed-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **RouteWise AI** is an enterprise-grade decision-support optimization prototype built for global logistics. It balances competing shipping objectives (**Cost**, **Transit Time**, and **Multi-Factor Risk**), enforces inviolable business constraints, provides full mathematical explainability for every recommendation, and benchmarks results against standard logistics heuristics.

---

## Table of Contents

1. [Problem Statement & Overview](#problem-statement--overview)
2. [Key Capabilities](#key-capabilities)
3. [Mathematical Formulation & Methodology](#mathematical-formulation--methodology)
4. [Empirical Evaluation & Baseline Benchmarks](#empirical-evaluation--baseline-benchmarks)
5. [System Architecture](#system-architecture)
6. [Repository Structure](#repository-structure)
7. [Installation & Quickstart](#installation--quickstart)
8. [Automated Pytest Suite](#automated-pytest-suite)
9. [Streamlit Dashboard Modules](#streamlit-dashboard-modules)
10. [Assumptions, Heuristics & Production Roadmap](#assumptions-heuristics--production-roadmap)

---

## Problem Statement & Overview

In international freight forwarding, selecting a shipping route is rarely a single-objective problem:
- **Cost minimization** often leads to protracted transit times, high transshipment delays, and vulnerable regional chokepoints.
- **Speed prioritization** dramatically inflates freight rates and burns operational margins.
- **Risk mitigation** demands premium routing around congested ports and geopolitical hotspots.

RouteWise AI provides an end-to-end multi-objective route recommendation engine with:
1. **Hard Constraint Filtering:** Non-negotiable exclusion of routes exceeding customer budget, transit deadlines, or delay thresholds.
2. **Min-Max Normalization:** Dimensionless scaling across route alternatives with mathematical division-by-zero protection.
3. **Configurable Weighted Scoring:** Flexible priority weighting reflecting business preferences.
4. **Transparent Explainability:** Plain-English and numeric breakdowns grounded exclusively in actual computed values.
5. **Sensitivity & Stability Analysis:** Simplex grid search mapping out recommendation stability across 66 weight combinations.
6. **Infeasible Route Diagnostics:** Closest-route recovery that quantifies required constraint relaxations when all alternatives fail.

---

## Mathematical Formulation & Methodology

### 1. Composite Risk Metric

Risk is synthesized from three empirical risk dimensions into a unified risk factor $R \in [0, 1]$:

$$R = 0.50 \cdot P(\text{delay}) + 0.30 \cdot R(\text{congestion}) + 0.20 \cdot R(\text{geopolitical})$$

- **Delay Probability ($0.50$):** Primary operational driver of customer SLA breaches.
- **Congestion Risk ($0.30$):** Port terminal queueing and berth dwell variability.
- **Geopolitical Risk ($0.20$):** Regional chokepoint vulnerabilities and canal transit volatility.

### 2. Relative Min-Max Normalization

To eliminate scale bias between currency ($\$), days, and unitless risk scores, metrics are normalized across available alternatives for a given shipment:

$$M_{\text{norm}} = \frac{M - M_{\min}}{M_{\max} - M_{\min}} \quad \text{if } M_{\max} > M_{\min} \text{ else } 0.0$$

*Safety Invariant:* When all candidate routes share identical values for a metric ($M_{\max} = M_{\min}$), the normalized metric evaluates safely to $0.0$, eliminating divide-by-zero singularities.

### 3. Weighted Composite Optimization Score

The overall optimization score for route $r$ is:

$$\text{Score}(r) = w_{\text{cost}} \cdot C_{\text{norm}}(r) + w_{\text{time}} \cdot T_{\text{norm}}(r) + w_{\text{risk}} \cdot R_{\text{norm}}(r)$$

$$\text{Subject to: } w_{\text{cost}}, w_{\text{time}}, w_{\text{risk}} \ge 0 \quad \text{and} \quad w_{\text{cost}} + w_{\text{time}} + w_{\text{risk}} = 1.0$$

**Lower scores denote superior routes.** Baseline weights are set to:
- $w_{\text{cost}} = 0.40$
- $w_{\text{time}} = 0.35$
- $w_{\text{risk}} = 0.25$

### 4. Pareto-Optimal Frontier Identification

In multi-objective optimization, a route $A$ **dominates** route $B$ ($A \succ B$) if and only if:
$$\text{Cost}(A) \le \text{Cost}(B) \land \text{Transit}(A) \le \text{Transit}(B) \land \text{Risk}(A) \le \text{Risk}(B)$$
with at least one strict inequality ($<$).

- **Non-Dominated Subset:** A route is **Pareto-optimal** if no other feasible route strictly dominates it across all three objectives.
- **Convex Scalarization Invariant:** For any weight vector with strictly positive weights ($w_{\text{cost}}, w_{\text{time}}, w_{\text{risk}} > 0$), the route that minimizes the aggregate linear score $\text{Score}(r)$ is mathematically guaranteed to belong to the Pareto-optimal frontier.

### 5. Hard Constraints, Soft Slack Advisories & Actionable Concessions

A route $r$ is **feasible** if and only if:
1. $\text{Cost}(r) \le \text{Max Budget}$
2. $\text{Transit Days}(r) \le \text{Max Transit Days}$
3. $\text{Delay Probability}(r) \le \text{Max Delay Probability}$

- **Soft Capacity Advisories:** When a feasible route operates with $<5\%$ remaining slack below any constraint limit (e.g. Budget Margin $< \$0.05 \cdot B_{\max}$), the system flags an operational advisory warning of high boundary sensitivity.
- **Normalized Violation Distance:** If all routes are infeasible, the diagnostic engine calculates:
  $$D_{\text{viol}}(r) = \max\left(0, \frac{\text{Cost} - B}{B}\right) + \max\left(0, \frac{\text{Time} - T}{T}\right) + \max\left(0, \frac{\text{Delay} - D}{D}\right)$$
- **Actionable Concessions:** For the closest infeasible candidate ($\arg\min D_{\text{viol}}$), exact numerical deficits and percentage relaxation concessions (e.g., `+12.5% Budget (+$450.00)`) are computed and presented to logistics dispatchers.

### 6. Configurable Priority Presets

Alongside freeform weight tuning, the platform provides calibrated operational priority profiles:
- **⚡ Critical (Speed & Risk First):** $w = (0.15, 0.55, 0.30)$ — Prioritizes high-value, perishable, or urgent cargo.
- **⏱️ Expedited (Transit Time Focus):** $w = (0.20, 0.60, 0.20)$ — Optimizes for tight SLA delivery windows.
- **⚖️ Balanced (Standard Logistics):** $w = (0.40, 0.35, 0.25)$ — Standard multimodal freight trade-off.
- **💰 Economy (Cost Minimization):** $w = (0.65, 0.20, 0.15)$ — Maximizes margin for flexible delivery schedules.

---

## Empirical Evaluation & Baseline Benchmarks

Evaluated over the full synthetic dataset of **40 global shipments** (153 total routes):

| Strategy | Shipments Evaluated | Feasible Shipments | Constraints Satisfied (%) | Avg Cost ($) | Avg Transit (days) | Avg Delay Prob | Avg Opt Score (lower = better) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weighted Optimization** | **40** | **34** | **85.0%** | **$9,312.09** | **23.9** | **16.5%** | **0.1808** |
| Cheapest Route | 40 | 34 | 85.0% | $8,380.06 | 26.4 | 19.1% | 0.2283 |
| Fastest Route | 40 | 34 | 85.0% | $9,796.56 | 22.9 | 16.6% | 0.2186 |
| Lowest-Risk Route | 40 | 34 | 85.0% | $9,696.31 | 24.3 | 14.8% | 0.2374 |

### Key Findings:
- **Optimal Multi-Attribute Utility:** Weighted Optimization achieves the lowest Average Optimization Score (**0.1808**), confirming that it successfully minimizes aggregate regret across conflicting objectives.
- **Cheapest Route Trade-Off:** The cheapest strategy saves $932 on average, but incurs a **2.5-day transit penalty** and a **15.8% increase in delay probability**.
- **Fastest Route Trade-Off:** The fastest strategy shaves 1.0 day off transit, but escalates costs by **+$484** per shipment.
- **Pareto Efficiency:** Weighted optimization strikes an optimal balance on the efficient frontier.

---

## Repository Structure

```
AIQON/
├── app.py                         # Streamlit multi-page executive application
├── generate_data.py               # Dataset generation runner script
├── generate_report.py             # 3-Page publication-grade PDF report generator
├── requirements.txt               # Application and test dependencies
├── README.md                      # Comprehensive project documentation
├── ARCHITECTURE.md                # System architecture and engineering specification
├── RouteWise_AI_Technical_Report.pdf # Generated 3-page technical report
├── data/
│   ├── shipments.csv              # 40 global shipments with business constraints
│   └── routes.csv                 # 153 route alternatives with multimodal metrics
├── src/
│   ├── __init__.py
│   ├── constraints.py             # Hard constraint engine & infeasibility diagnostics
│   ├── data_generator.py          # Synthetic dataset generator with 15 global hubs
│   ├── data_loader.py             # Data loading, validation, and schema assertions
│   ├── evaluation.py              # Baseline strategy benchmark evaluator
│   ├── explainability.py          # Value-grounded explanation and trade-off generator
│   ├── optimizer.py               # Min-max normalization and weighted-sum optimizer
│   ├── preprocessing.py           # Feature engineering and composite risk calculator
│   ├── recommender.py             # Recommendation orchestration pipeline
│   └── sensitivity.py             # Simplex weight search and constraint elasticity
└── tests/
    ├── test_constraints.py        # Constraint enforcement and violation margin tests
    ├── test_data_generator.py     # Dataset distribution, schema, and loader tests
    ├── test_evaluation.py         # Baseline strategy and full evaluation tests
    ├── test_explainability.py     # Explanation formatting and impact contribution tests
    ├── test_optimizer.py          # Weights validation, normalization, and scoring tests
    ├── test_preprocessing.py     # Risk classification and hub parsing tests
    ├── test_recommender.py        # Pipeline invariants and recommendation tests
    └── test_sensitivity.py        # Weight simplex search and constraint margin tests
```

---

## Installation & Quickstart

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 2. Environment Setup

```bash
# Clone or navigate to the repository
cd AIQON

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Synthetic Dataset

```bash
python generate_data.py
```
*Output: Generates 40 shipments and 153 alternative routes into `data/shipments.csv` and `data/routes.csv`.*

### 4. Launch Streamlit Application

```bash
streamlit run app.py
```
*The interactive dashboard will open automatically at `http://localhost:8501`.*

### 5. Generate Technical PDF Report

```bash
python generate_report.py
```
*Output: Generates `RouteWise_AI_Technical_Report.pdf` (3-page formal report).*

---

## Automated Pytest Suite

RouteWise AI comes with **88 comprehensive automated unit and integration tests**:

```bash
pytest -v
```

### Test Coverage Highlights:
- **Constraints (`test_constraints.py`):** Individual constraint checks, boundary conditions, multiple violations, margin calculations, and closest-infeasible recovery.
- **Optimizer (`test_optimizer.py`):** Strict weight validation (sum to 1.0, non-negative, numeric type checking), min-max scaling, division-by-zero handling, and route ranking consistency.
- **Recommender (`test_recommender.py`):** End-to-end recommendation pipeline, explanation completeness, score breakdown presence, constraint overrides, and bulk invariance.
- **Evaluation (`test_evaluation.py`):** Benchmark calculations for Cheapest, Fastest, Lowest-Risk, and Weighted Optimization strategies.
- **Sensitivity (`test_sensitivity.py`):** 66-combination simplex exploration, route switching detection, and constraint elasticity.
- **Data Generator (`test_data_generator.py`):** Shape assertions (30-50 shipments, 3-5 routes each), seed reproducibility, and schema validation.
- **Preprocessing & Explainability (`test_preprocessing.py`, `test_explainability.py`):** Composite risk arithmetic, risk tiers, hub string parsing, trade-off summaries, and percentage contributions.

---

## Streamlit Dashboard Modules

The application features 6 specialized executive views:

1. **📊 Executive Dashboard:**
   - Network KPIs: Shipments, evaluated routes, average cost, transit days, and delay probability.
   - Cost and transit distributions, risk tier breakdown pie chart, cost vs. transit scatter plot.
   - Priority breakdowns and risk box plots across delay, congestion, and geopolitical indicators.

2. **📦 Shipment Management:**
   - Multi-criteria filtering (Origin, Destination, Priority, ID search).
   - Detailed route alternatives inspection per shipment.
   - Interactive **Add New Shipment** form to add custom shipments and route alternatives to the active network with full persistence.

3. **🔀 Route Comparison:**
   - Side-by-side tabular comparison of alternatives with status badges.
   - Multi-objective Radar chart with star indicator on the recommended route.
   - Cost and transit bar charts with customer limit threshold overlay lines.

4. **🎯 Recommendation Center:**
   - Recommended route KPI summary with score breakdown.
   - Weighted score contribution horizontal bar chart and composition donut chart.
   - Value-grounded plain-language explanation and comparative trade-off summary.
   - Real-time constraint override modeling (custom budget, transit, or delay thresholds).
   - CSV export of route analysis.

5. **📈 Sensitivity Analysis:**
   - 66-combination weight simplex search showing route switching frequencies.
   - 2D Heatmap of recommended route IDs across Cost vs. Time weights.
   - Multi-tab constraint elasticity curves (Budget, Transit, Delay probability).

6. **📋 Evaluation & Baselines:**
   - Comprehensive benchmark comparison table across all 4 strategies.
   - Bar charts comparing average cost, transit time, delay probability, and optimization score.
   - Strategy Trade-off Radar comparing all four strategies across competing dimensions.
   - CSV export of benchmark results.

---

## Assumptions, Heuristics & Production Roadmap

### Model Assumptions & Heuristic Boundaries:
1. **Decision Support Heuristic:** Weighted-sum optimization is a transparent decision-support tool. It assumes linear trade-offs between normalized objectives.
2. **Relative Normalization Scope:** Normalization bounds are computed locally per shipment alternative set. Raw scores should not be compared directly across different corridors.
3. **Synthetic Ground Truth:** Synthetic data simulates realistic logistics distributions across 15 global hubs (Shanghai, Singapore, Rotterdam, Los Angeles, Dubai, Hamburg, Mumbai, etc.) but does not represent live operational telemetry.

### Production Roadmap:
- **Phase 1: Live Telemetry Ingestion:** Integrate AIS maritime vessel tracking feeds, port terminal API wait-times, and satellite weather alerts.
- **Phase 2: Non-Linear & Pareto Optimization:** Implement NSGA-II multi-objective genetic algorithms and epsilon-constraint solvers for non-convex Pareto frontiers.
- **Phase 3: Dynamic Carrier Contract Integration:** Incorporate tiered volume commitments, detention & demurrage tariffs, and dynamic fuel surcharge (BAF) hedging.

---

## License

MIT License. Developed for the AIQON AI Internship Challenge — October 2026.
