"""
RouteWise AI — Intelligent Shipment Route Optimization
=======================================================

Main Streamlit application entry point.

A multi-objective route optimization system for global logistics
that evaluates shipment routes across cost, transit time, and risk,
recommends optimal routes, and explains decisions transparently.

All data is synthetic and generated for demonstration purposes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_data
from src.preprocessing import preprocess_routes, compute_hub_count
from src.optimizer import OptimizationWeights, optimize_routes, get_priority_weights
from src.constraints import (
    evaluate_shipment_constraints,
    get_feasible_routes,
    find_closest_infeasible,
    compute_required_relaxations,
)
from src.recommender import recommend_route, recommend_all_shipments, Recommendation
from src.explainability import (
    generate_score_comparison_table,
    explain_weight_impact,
    generate_trade_off_summary,
    format_constraint_status,
)
from src.sensitivity import (
    weight_sensitivity_analysis,
    constraint_sensitivity,
    summarize_weight_sensitivity,
)
from src.evaluation import (
    run_full_evaluation,
    evaluation_to_dataframe,
)

# ─────────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RouteWise AI — Intelligent Shipment Route Optimization",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  Custom Styling (Executive Dark Theme)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #091322 0%, #112240 50%, #1e3a5f 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        color: white;
        font-family: 'Inter', sans-serif;
        border: 1px solid #1e3a5f;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
    }
    .app-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.92rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* KPI Card styling - dark glassmorphism */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.90) 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem 0.9rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        border-color: #38bdf8;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
        font-family: 'Inter', sans-serif;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0.35rem 0 0 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-family: 'Inter', sans-serif;
    }

    /* Risk badges */
    .risk-low { color: #34d399; background: rgba(5, 150, 105, 0.2); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; border: 1px solid rgba(52, 211, 153, 0.3); }
    .risk-medium { color: #fbbf24; background: rgba(217, 119, 6, 0.2); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; border: 1px solid rgba(251, 191, 36, 0.3); }
    .risk-high { color: #f87171; background: rgba(220, 38, 38, 0.2); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; border: 1px solid rgba(248, 113, 113, 0.3); }
    .risk-critical { color: #fda4af; background: rgba(159, 18, 57, 0.25); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; border: 1px solid rgba(253, 164, 175, 0.3); }

    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.45rem;
        margin-bottom: 1rem;
        margin-top: 0.6rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091322 0%, #112240 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Status badges */
    .recommended-badge {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .pareto-badge {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 3px 9px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
        margin-left: 6px;
    }
    .infeasible-badge {
        background: rgba(220, 38, 38, 0.2);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    /* Disclaimer box */
    .data-disclaimer {
        background: rgba(30, 41, 59, 0.55);
        border-left: 3px solid #38bdf8;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 14px;
        font-size: 0.83rem;
        color: #94a3b8;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Data Loading (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    """Load and preprocess all data."""
    shipments_df, routes_df = get_data()
    routes_processed = preprocess_routes(routes_df)
    return shipments_df, routes_df, routes_processed


@st.cache_data
def run_evaluation_cached(_shipments_df, _routes_df):
    """Run full evaluation (cached)."""
    return run_full_evaluation(_shipments_df, _routes_df)


# Load data
try:
    shipments_df, routes_df, routes_processed = load_all_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.info("Run `python -c \"from src.data_generator import generate_dataset; generate_dataset()\"` to generate data.")
    st.stop()


# ─────────────────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚢 RouteWise AI")
    st.markdown("*Intelligent Route Optimization*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "📦 Shipment Management",
            "🔀 Route Comparison",
            "🎯 Recommendation Center",
            "📈 Sensitivity Analysis",
            "📋 Evaluation & Baselines",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Global weight controls
    st.markdown("### ⚖️ Optimization Weights")
    st.caption("Adjust to reflect shipping priorities")

    preset_options = [
        "⚖️ Balanced (Standard)",
        "⚡ Critical (Speed & Risk)",
        "⏱️ Expedited (Transit Time)",
        "💰 Economy (Cost Focus)",
        "⚙️ Custom",
    ]

    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = preset_options[0]

    chosen_preset = st.selectbox(
        "Priority Profile Preset",
        preset_options,
        index=preset_options.index(st.session_state.selected_preset),
        key="preset_select",
    )

    if chosen_preset != st.session_state.selected_preset:
        st.session_state.selected_preset = chosen_preset
        preset_mapping = {
            "⚖️ Balanced (Standard)": "Medium",
            "⚡ Critical (Speed & Risk)": "Critical",
            "⏱️ Expedited (Transit Time)": "High",
            "💰 Economy (Cost Focus)": "Low",
        }
        if chosen_preset in preset_mapping:
            pw = get_priority_weights(preset_mapping[chosen_preset])
            st.session_state.global_w_cost = pw.cost
            st.session_state.global_w_time = pw.time
            st.rerun()

    w_cost = st.slider("Cost Weight", 0.0, 1.0, 0.40, 0.05, key="global_w_cost")
    w_time = st.slider("Time Weight", 0.0, 1.0, 0.35, 0.05, key="global_w_time")

    if w_cost + w_time > 1.0:
        st.warning("⚠️ Cost + Time exceeds 1.0. Auto-adjusting to maintain valid weights.")
        remaining = max(0.0, round(1.0 - w_cost, 2))
        w_time = min(w_time, remaining)
        w_risk = round(1.0 - w_cost - w_time, 2)
    else:
        w_risk = round(1.0 - w_cost - w_time, 2)
        w_risk = max(0.0, min(1.0, w_risk))

    # Guard against float rounding
    total_weights = round(w_cost + w_time + w_risk, 6)
    if total_weights != 1.0:
        w_risk = round(1.0 - w_cost - w_time, 2)

    st.metric("Risk Weight (auto)", f"{w_risk:.2f}")

    try:
        weights = OptimizationWeights(cost=w_cost, time=w_time, risk=w_risk)
    except (ValueError, TypeError) as e:
        weights = OptimizationWeights(cost=0.40, time=0.35, risk=0.25)
        st.info("Using default weights (0.40, 0.35, 0.25).")

    st.markdown("---")
    st.caption(f"📊 {len(shipments_df)} shipments | {len(routes_df)} routes")
    st.caption("Operational synthetic telemetry")


# ─────────────────────────────────────────────────────────
#  Color scheme helper & Dark Chart Theming
# ─────────────────────────────────────────────────────────
COLORS = {
    "primary": "#38bdf8",
    "secondary": "#60a5fa",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "info": "#818cf8",
    "light": "#f1f5f9",
    "chart_palette": ["#38bdf8", "#60a5fa", "#34d399", "#818cf8", "#fbbf24", "#f87171"],
}

RISK_COLORS = {
    "Low": "#34d399",
    "Medium": "#fbbf24",
    "High": "#f87171",
    "Critical": "#fda4af",
}


def apply_chart_theme(fig, title=None, x_title=None, y_title=None, height=None, showlegend=True):
    """Apply consistent modern dark theme to Plotly figures."""
    layout_updates = dict(
        paper_bgcolor="rgba(15, 23, 42, 0.75)",
        plot_bgcolor="rgba(30, 41, 59, 0.50)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        title_font=dict(family="Inter, sans-serif", color="#f8fafc", size=14),
        legend=dict(
            font=dict(color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.65)",
            bordercolor="rgba(51, 65, 85, 0.5)",
            borderwidth=1,
        ),
        margin=dict(t=50 if title else 30, b=45, l=55, r=25),
    )
    if title:
        layout_updates["title"] = dict(text=title, x=0.02, xanchor="left")
    if height:
        layout_updates["height"] = height
    if showlegend is not None:
        layout_updates["showlegend"] = showlegend

    fig.update_layout(**layout_updates)
    fig.update_xaxes(
        gridcolor="rgba(51, 65, 85, 0.4)",
        zerolinecolor="rgba(51, 65, 85, 0.6)",
        tickfont=dict(color="#94a3b8"),
        title_font=dict(color="#e2e8f0"),
    )
    fig.update_yaxes(
        gridcolor="rgba(51, 65, 85, 0.4)",
        zerolinecolor="rgba(51, 65, 85, 0.6)",
        tickfont=dict(color="#94a3b8"),
        title_font=dict(color="#e2e8f0"),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def render_kpi(value, label, prefix="", suffix=""):
    """Render a KPI card."""
    display_val = f"{prefix}{value}{suffix}"
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value">{display_val}</p>
        <p class="kpi-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  MODULE 1: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────
if page == "📊 Executive Dashboard":
    st.markdown(f"""
    <div class="app-header">
        <h1>📊 Executive Dashboard</h1>
        <p>Real-time overview of shipment optimization performance across your logistics network</p>
    </div>
    <div class="data-disclaimer">
        ℹ️ <b>Operational Simulation Mode:</b> Evaluated on synthetic multimodal logistics telemetry
        ({len(shipments_df)} shipments, {len(routes_df)} route alternatives).
        Multi-objective optimization scores, Pareto efficiency frontiers, and constraint slacks are evaluated dynamically in real time.
    </div>
    """, unsafe_allow_html=True)

    # Run optimization for all shipments to compute dashboard stats
    recommendations = recommend_all_shipments(shipments_df, routes_df, weights)

    feasible_recs = [r for r in recommendations if r.has_feasible_routes]
    no_feasible_recs = [r for r in recommendations if not r.has_feasible_routes]

    # Get all feasible routes count
    all_feasible_count = 0
    for rec in recommendations:
        all_feasible_count += rec.feasible_count

    # KPI Row 1
    st.markdown('<p class="section-header">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi(len(shipments_df), "Total Shipments")
    with c2:
        render_kpi(len(routes_df), "Routes Evaluated")
    with c3:
        render_kpi(f"${routes_processed['shipping_cost'].mean():,.0f}", "Avg Shipping Cost")
    with c4:
        render_kpi(f"{routes_processed['transit_days'].mean():.1f}", "Avg Transit Days")

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_kpi(f"{routes_processed['delay_probability'].mean():.1%}", "Avg Delay Probability")
    with c6:
        render_kpi(all_feasible_count, "Feasible Routes")
    with c7:
        render_kpi(len(no_feasible_recs), "No Feasible Route", suffix=" shipments")
    with c8:
        if feasible_recs:
            avg_score = np.mean([r.score_breakdown["total_score"] for r in feasible_recs if r.score_breakdown])
            render_kpi(f"{avg_score:.3f}", "Avg Optimization Score")
        else:
            render_kpi("N/A", "Avg Optimization Score")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    st.markdown('<p class="section-header">Distribution Analysis</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            routes_processed, x="shipping_cost", nbins=30,
            color_discrete_sequence=[COLORS["secondary"]],
        )
        apply_chart_theme(fig, title="Shipping Cost Distribution", x_title="Shipping Cost ($)", y_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            routes_processed, x="transit_days", nbins=25,
            color_discrete_sequence=[COLORS["info"]],
        )
        apply_chart_theme(fig, title="Transit Time Distribution", x_title="Transit Days", y_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        risk_counts = routes_processed["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.pie(
            risk_counts, names="Risk Level", values="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )
        apply_chart_theme(fig, title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            routes_processed,
            x="shipping_cost", y="transit_days",
            color="risk_level",
            size="delay_probability",
            color_discrete_map=RISK_COLORS,
            hover_data=["route_id", "shipment_id"],
        )
        apply_chart_theme(fig, title="Cost vs Transit Time (sized by delay probability)", x_title="Shipping Cost ($)", y_title="Transit Days")
        st.plotly_chart(fig, use_container_width=True)

    # Route selection distribution
    st.markdown('<p class="section-header">Recommendation Overview</p>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        # Priority distribution
        priority_counts = shipments_df["priority"].value_counts().reset_index()
        priority_counts.columns = ["Priority", "Count"]
        fig = px.bar(
            priority_counts, x="Priority", y="Count",
            color="Priority",
            color_discrete_map={
                "Critical": "#f87171", "High": "#fbbf24",
                "Medium": "#60a5fa", "Low": "#34d399",
            },
        )
        apply_chart_theme(fig, title="Shipments by Priority", x_title="Priority", y_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        # Routes per shipment
        routes_per_shipment = routes_df.groupby("shipment_id").size().reset_index(name="Routes")
        fig = px.histogram(
            routes_per_shipment, x="Routes", nbins=5,
            color_discrete_sequence=[COLORS["primary"]],
        )
        apply_chart_theme(fig, title="Route Alternatives per Shipment", x_title="Number of Routes", y_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Risk comparison chart
    st.markdown('<p class="section-header">Risk Comparison</p>', unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Delay Probability", "Congestion Risk", "Geopolitical Risk"))

    fig.add_trace(
        go.Box(y=routes_processed["delay_probability"], name="Delay", marker_color=COLORS["secondary"]),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=routes_processed["congestion_risk"], name="Congestion", marker_color=COLORS["warning"]),
        row=1, col=2,
    )
    fig.add_trace(
        go.Box(y=routes_processed["geopolitical_risk"], name="Geopolitical", marker_color=COLORS["danger"]),
        row=1, col=3,
    )
    apply_chart_theme(fig, title="Risk Variance Across Logistics Network", height=350, showlegend=False)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#f8fafc', size=13, family='Inter, sans-serif')
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  MODULE 2: SHIPMENT MANAGEMENT
# ─────────────────────────────────────────────────────────
elif page == "📦 Shipment Management":
    st.markdown("""
    <div class="app-header">
        <h1>📦 Shipment Management</h1>
        <p>Browse, filter, and manage shipment records and their route alternatives</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    st.markdown('<p class="section-header">Filters</p>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        origin_filter = st.multiselect("Origin", sorted(shipments_df["origin"].unique()))
    with fc2:
        dest_filter = st.multiselect("Destination", sorted(shipments_df["destination"].unique()))
    with fc3:
        priority_filter = st.multiselect("Priority", ["Critical", "High", "Medium", "Low"])
    with fc4:
        shipment_search = st.text_input("Shipment ID", placeholder="e.g. SH-001")

    # Apply filters
    filtered = shipments_df.copy()
    if origin_filter:
        filtered = filtered[filtered["origin"].isin(origin_filter)]
    if dest_filter:
        filtered = filtered[filtered["destination"].isin(dest_filter)]
    if priority_filter:
        filtered = filtered[filtered["priority"].isin(priority_filter)]
    if shipment_search:
        filtered = filtered[filtered["shipment_id"].str.contains(shipment_search, case=False)]

    st.markdown(f"**Showing {len(filtered)} of {len(shipments_df)} shipments**")

    # Shipments table
    display_df = filtered.copy()
    display_df["max_budget"] = display_df["max_budget"].apply(lambda x: f"${x:,.2f}")
    display_df["max_delay_probability"] = display_df["max_delay_probability"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Shipment ID", "Origin", "Destination", "Priority",
                          "Max Budget", "Max Transit Days", "Max Delay Prob"]

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    # Shipment detail inspection
    st.markdown("---")
    st.markdown('<p class="section-header">Shipment Details</p>', unsafe_allow_html=True)

    selected_shipment_id = st.selectbox(
        "Select shipment to inspect",
        filtered["shipment_id"].tolist(),
        key="mgmt_shipment_select",
    )

    if selected_shipment_id:
        shipment = shipments_df[shipments_df["shipment_id"] == selected_shipment_id].iloc[0]
        ship_routes = routes_processed[routes_processed["shipment_id"] == selected_shipment_id]

        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.markdown(f"**Origin:** {shipment['origin']}")
            st.markdown(f"**Destination:** {shipment['destination']}")
        with dc2:
            st.markdown(f"**Priority:** {shipment['priority']}")
            st.markdown(f"**Route Alternatives:** {len(ship_routes)}")
        with dc3:
            st.markdown(f"**Max Budget:** ${shipment['max_budget']:,.2f}")
            st.markdown(f"**Max Transit:** {shipment['max_transit_days']:.1f} days")
        with dc4:
            st.markdown(f"**Max Delay Prob:** {shipment['max_delay_probability']:.1%}")

        # Show routes for this shipment
        st.markdown("**Route Alternatives:**")
        route_display = ship_routes[["route_id", "intermediate_hubs", "shipping_cost",
                                      "transit_days", "delay_probability", "congestion_risk",
                                      "geopolitical_risk", "risk_level"]].copy()
        route_display.columns = ["Route ID", "Intermediate Hubs", "Cost ($)", "Transit (days)",
                                  "Delay Prob", "Congestion Risk", "Geo Risk", "Risk Level"]
        st.dataframe(route_display, use_container_width=True, hide_index=True)

    # Add new shipment form
    st.markdown("---")
    st.markdown('<p class="section-header">➕ Add New Shipment</p>', unsafe_allow_html=True)

    with st.expander("Create a new shipment with route alternatives", expanded=False):
        with st.form("new_shipment_form"):
            nf1, nf2 = st.columns(2)
            with nf1:
                new_origin = st.selectbox("Origin", sorted(shipments_df["origin"].unique()), key="new_origin")
                new_priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"], key="new_priority")
                new_budget = st.number_input("Max Budget ($)", min_value=100.0, value=5000.0, step=100.0)
            with nf2:
                new_destination = st.selectbox("Destination", sorted(shipments_df["destination"].unique()), key="new_dest")
                new_transit = st.number_input("Max Transit Days", min_value=1.0, value=20.0, step=1.0)
                new_delay = st.number_input("Max Delay Probability", min_value=0.01, max_value=1.0, value=0.30, step=0.01)

            st.markdown("**Route Alternatives** (add at least 3)")
            route_entries = []
            for r_idx in range(3):
                st.markdown(f"*Route {r_idx + 1}:*")
                rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                with rc1:
                    r_cost = st.number_input(f"Cost ($)", min_value=100.0, value=3000.0 + r_idx * 1000, step=100.0, key=f"rc_{r_idx}")
                with rc2:
                    r_transit = st.number_input(f"Transit (days)", min_value=1.0, value=10.0 + r_idx * 5, step=1.0, key=f"rt_{r_idx}")
                with rc3:
                    r_delay = st.number_input(f"Delay Prob", min_value=0.01, max_value=1.0, value=0.1 + r_idx * 0.05, step=0.01, key=f"rd_{r_idx}")
                with rc4:
                    r_cong = st.number_input(f"Congestion", min_value=0.01, max_value=1.0, value=0.15 + r_idx * 0.1, step=0.01, key=f"rcg_{r_idx}")
                with rc5:
                    r_geo = st.number_input(f"Geo Risk", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key=f"rg_{r_idx}")
                route_entries.append({"cost": r_cost, "transit": r_transit, "delay": r_delay, "cong": r_cong, "geo": r_geo})

            submitted = st.form_submit_button("Add Shipment", type="primary")

            if submitted:
                if new_origin == new_destination:
                    st.error("Origin and destination must be different.")
                elif new_budget <= 0 or new_transit <= 0 or new_delay <= 0:
                    st.error("All constraints must be positive.")
                else:
                    # Generate new shipment ID
                    existing_ids = shipments_df["shipment_id"].tolist()
                    max_id_num = max(int(sid.split("-")[1]) for sid in existing_ids)
                    new_id = f"SH-{max_id_num + 1:03d}"

                    new_shipment = pd.DataFrame([{
                        "shipment_id": new_id,
                        "origin": new_origin,
                        "destination": new_destination,
                        "priority": new_priority,
                        "max_budget": new_budget,
                        "max_transit_days": new_transit,
                        "max_delay_probability": new_delay,
                    }])

                    new_routes = []
                    for r_idx, entry in enumerate(route_entries):
                        new_routes.append({
                            "shipment_id": new_id,
                            "route_id": f"{new_id}-R{r_idx + 1:02d}",
                            "intermediate_hubs": "User-defined",
                            "shipping_cost": entry["cost"],
                            "transit_days": entry["transit"],
                            "delay_probability": entry["delay"],
                            "congestion_risk": entry["cong"],
                            "geopolitical_risk": entry["geo"],
                        })
                    new_routes_df = pd.DataFrame(new_routes)

                    # Save to CSV (append)
                    data_dir = "data"
                    shipments_path = os.path.join(data_dir, "shipments.csv")
                    routes_path = os.path.join(data_dir, "routes.csv")

                    new_shipment.to_csv(shipments_path, mode="a", header=False, index=False)
                    new_routes_df.to_csv(routes_path, mode="a", header=False, index=False)

                    st.success(f"✅ Shipment **{new_id}** added successfully with {len(route_entries)} routes!")
                    st.cache_data.clear()
                    st.rerun()


# ─────────────────────────────────────────────────────────
#  MODULE 3: ROUTE COMPARISON
# ─────────────────────────────────────────────────────────
elif page == "🔀 Route Comparison":
    st.markdown("""
    <div class="app-header">
        <h1>🔀 Route Comparison</h1>
        <p>Compare route alternatives and identify the optimal shipping path</p>
    </div>
    """, unsafe_allow_html=True)

    # Shipment selection
    selected_sid = st.selectbox(
        "Select Shipment",
        shipments_df["shipment_id"].tolist(),
        format_func=lambda x: f"{x} ({shipments_df[shipments_df['shipment_id']==x].iloc[0]['origin']} → {shipments_df[shipments_df['shipment_id']==x].iloc[0]['destination']})",
        key="compare_shipment",
    )

    shipment = shipments_df[shipments_df["shipment_id"] == selected_sid].iloc[0]
    ship_routes = routes_processed[routes_processed["shipment_id"] == selected_sid].copy()

    # Shipment info
    info_cols = st.columns(5)
    with info_cols[0]:
        st.metric("Origin", shipment["origin"])
    with info_cols[1]:
        st.metric("Destination", shipment["destination"])
    with info_cols[2]:
        st.metric("Priority", shipment["priority"])
    with info_cols[3]:
        st.metric("Budget Limit", f"${shipment['max_budget']:,.2f}")
    with info_cols[4]:
        st.metric("Transit Limit", f"{shipment['max_transit_days']:.1f} days")

    st.markdown("---")

    # Get recommendation
    rec = recommend_route(shipment, routes_df[routes_df["shipment_id"] == selected_sid], weights)

    # Constraint status
    st.markdown('<p class="section-header">Constraint Evaluation</p>', unsafe_allow_html=True)
    constraint_table = format_constraint_status(rec.constraint_results)
    st.dataframe(constraint_table, use_container_width=True, hide_index=True)

    # Sorting options
    st.markdown('<p class="section-header">Route Comparison</p>', unsafe_allow_html=True)
    sort_option = st.selectbox(
        "Sort by",
        ["Overall Optimization Score", "Lowest Cost", "Fastest Transit", "Lowest Risk"],
        key="sort_routes",
    )

    # Score the routes
    scored_routes = rec.all_routes_scored.copy()

    sort_map = {
        "Lowest Cost": "shipping_cost",
        "Fastest Transit": "transit_days",
        "Lowest Risk": "composite_risk",
        "Overall Optimization Score": "optimization_score",
    }
    sort_col = sort_map[sort_option]
    if sort_col in scored_routes.columns:
        scored_routes = scored_routes.sort_values(sort_col, na_position="last")

    # Display comparison table
    display_cols = ["route_id", "intermediate_hubs", "shipping_cost", "transit_days",
                    "delay_probability", "congestion_risk", "geopolitical_risk",
                    "reliability_score", "risk_level", "is_feasible", "is_pareto_optimal"]
    if "optimization_score" in scored_routes.columns:
        display_cols.extend(["optimization_score", "rank"])

    available = [c for c in display_cols if c in scored_routes.columns]
    comparison_df = scored_routes[available].copy()

    # Rename for display
    rename_map = {
        "route_id": "Route ID", "intermediate_hubs": "Intermediate Hubs",
        "shipping_cost": "Cost ($)", "transit_days": "Transit (days)",
        "delay_probability": "Delay Prob", "congestion_risk": "Congestion",
        "geopolitical_risk": "Geo Risk", "reliability_score": "Reliability",
        "risk_level": "Risk Level", "is_feasible": "Feasible",
        "is_pareto_optimal": "Pareto Optimal",
        "optimization_score": "Opt Score", "rank": "Rank",
    }
    comparison_df = comparison_df.rename(columns=rename_map)
    if "Pareto Optimal" in comparison_df.columns:
        comparison_df["Pareto Optimal"] = comparison_df["Pareto Optimal"].apply(lambda x: "⭐ Yes" if x else "Dominated")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Highlight recommended route
    if rec.has_feasible_routes and rec.recommended_route_id:
        pareto_badge = '<span class="pareto-badge">⭐ Pareto Optimal</span>' if getattr(rec, "is_pareto_optimal", True) else ''
        st.markdown(f'<span class="recommended-badge">✅ Recommended: {rec.recommended_route_id}</span> {pareto_badge}',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="infeasible-badge">⚠️ No feasible route found</span>',
                    unsafe_allow_html=True)

    # Visual comparisons
    st.markdown("---")
    chart_cols = st.columns(2)

    with chart_cols[0]:
        fig = px.bar(
            ship_routes.sort_values("shipping_cost"),
            x="route_id", y="shipping_cost",
            color="risk_level",
            labels={"route_id": "Route", "shipping_cost": "Cost ($)", "risk_level": "Risk"},
            color_discrete_map=RISK_COLORS,
        )
        if rec.recommended_route_id:
            fig.add_hline(y=shipment["max_budget"], line_dash="dash", line_color="#ef4444",
                         annotation_text="Budget Limit", annotation_font_color="#f87171")
        apply_chart_theme(fig, title="Cost by Route", x_title="Route", y_title="Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        fig = px.bar(
            ship_routes.sort_values("transit_days"),
            x="route_id", y="transit_days",
            color="risk_level",
            labels={"route_id": "Route", "transit_days": "Transit (days)", "risk_level": "Risk"},
            color_discrete_map=RISK_COLORS,
        )
        fig.add_hline(y=shipment["max_transit_days"], line_dash="dash", line_color="#ef4444",
                     annotation_text="Transit Limit", annotation_font_color="#f87171")
        apply_chart_theme(fig, title="Transit Time by Route", x_title="Route", y_title="Transit (days)")
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart for multi-objective comparison
    if len(ship_routes) > 0:
        st.markdown('<p class="section-header">Multi-Objective Comparison</p>', unsafe_allow_html=True)

        categories = ["Cost", "Transit Time", "Delay Prob", "Congestion", "Geo Risk"]
        fig = go.Figure()

        for _, route in ship_routes.iterrows():
            # Normalize each metric to 0-1 for radar chart
            vals = [
                route["shipping_cost"] / ship_routes["shipping_cost"].max() if ship_routes["shipping_cost"].max() > 0 else 0,
                route["transit_days"] / ship_routes["transit_days"].max() if ship_routes["transit_days"].max() > 0 else 0,
                route["delay_probability"],
                route["congestion_risk"],
                route["geopolitical_risk"],
            ]
            name = route["route_id"]
            if rec.recommended_route_id and route["route_id"] == rec.recommended_route_id:
                name += " ⭐"

            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=name,
                fill="toself",
                opacity=0.6,
            ))

        fig.update_layout(
            polar=dict(
                bgcolor="rgba(30, 41, 59, 0.4)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor="rgba(51, 65, 85, 0.5)",
                    tickfont=dict(color="#94a3b8"),
                    linecolor="rgba(51, 65, 85, 0.6)",
                ),
                angularaxis=dict(
                    gridcolor="rgba(51, 65, 85, 0.5)",
                    tickfont=dict(color="#cbd5e1"),
                    linecolor="rgba(51, 65, 85, 0.6)",
                ),
            ),
            paper_bgcolor="rgba(15, 23, 42, 0.75)",
            plot_bgcolor="rgba(30, 41, 59, 0.50)",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            title_font=dict(color="#f8fafc", size=14),
            legend=dict(
                font=dict(color="#cbd5e1"),
                bgcolor="rgba(15, 23, 42, 0.65)",
                bordercolor="rgba(51, 65, 85, 0.5)",
            ),
            title="Route Characteristics Radar (lower = better)",
            height=450,
            margin=dict(t=50, b=30, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  MODULE 4 & 5: RECOMMENDATION CENTER
# ─────────────────────────────────────────────────────────
elif page == "🎯 Recommendation Center":
    st.markdown("""
    <div class="app-header">
        <h1>🎯 Recommendation Center</h1>
        <p>AI-powered route recommendations with full explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # Shipment selection
    selected_sid = st.selectbox(
        "Select Shipment for Recommendation",
        shipments_df["shipment_id"].tolist(),
        format_func=lambda x: f"{x} — {shipments_df[shipments_df['shipment_id']==x].iloc[0]['origin']} → {shipments_df[shipments_df['shipment_id']==x].iloc[0]['destination']} ({shipments_df[shipments_df['shipment_id']==x].iloc[0]['priority']})",
        key="rec_shipment",
    )

    shipment = shipments_df[shipments_df["shipment_id"] == selected_sid].iloc[0]
    ship_routes_raw = routes_df[routes_df["shipment_id"] == selected_sid].copy()

    # Custom constraint overrides
    with st.expander("⚙️ Override Constraints (optional)", expanded=False):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            custom_budget = st.number_input(
                "Budget Override ($)",
                value=float(shipment["max_budget"]),
                min_value=0.0,
                step=100.0,
            )
        with oc2:
            custom_transit = st.number_input(
                "Transit Override (days)",
                value=float(shipment["max_transit_days"]),
                min_value=0.0,
                step=1.0,
            )
        with oc3:
            custom_delay = st.number_input(
                "Delay Override",
                value=float(shipment["max_delay_probability"]),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )

    # Generate recommendation
    rec = recommend_route(
        shipment, ship_routes_raw, weights,
        custom_budget=custom_budget if custom_budget != shipment["max_budget"] else None,
        custom_transit=custom_transit if custom_transit != shipment["max_transit_days"] else None,
        custom_delay=custom_delay if custom_delay != shipment["max_delay_probability"] else None,
    )

    st.markdown("---")

    if rec.has_feasible_routes and rec.recommended_route is not None:
        # ── Recommended route card ──
        st.markdown('<p class="section-header">🏆 Recommended Route</p>', unsafe_allow_html=True)

        best = rec.recommended_route

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            render_kpi(rec.recommended_route_id, "Route ID")
        with mc2:
            render_kpi(f"${best['shipping_cost']:,.2f}", "Cost")
        with mc3:
            render_kpi(f"{best['transit_days']:.1f}", "Transit Days")
        with mc4:
            render_kpi(f"{best['delay_probability']:.1%}", "Delay Prob")
        with mc5:
            pareto_str = " (⭐ Pareto)" if getattr(rec, "is_pareto_optimal", True) else ""
            render_kpi(f"{rec.score_breakdown['total_score']:.4f}", f"Opt Score{pareto_str}")

        st.markdown("<br>", unsafe_allow_html=True)

        mc6, mc7, mc8 = st.columns(3)
        with mc6:
            risk_level = best.get("risk_level", "Unknown")
            st.markdown(f"**Risk Level:** {risk_level}")
            st.markdown(f"**Intermediate Hubs:** {best['intermediate_hubs']}")
        with mc7:
            st.markdown(f"**Feasible Routes:** {rec.feasible_count} / {rec.total_routes}")
            st.markdown(f"**Congestion Risk:** {best['congestion_risk']:.3f}")
        with mc8:
            st.markdown(f"**Geopolitical Risk:** {best['geopolitical_risk']:.3f}")
            st.markdown(f"**Reliability:** {best.get('reliability_score', 'N/A')}")

        # Score breakdown visualization
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Score Breakdown</p>', unsafe_allow_html=True)

        sb = rec.score_breakdown
        breakdown_cols = st.columns(2)

        with breakdown_cols[0]:
            fig = go.Figure(go.Bar(
                x=[sb["weighted_cost"], sb["weighted_time"], sb["weighted_risk"]],
                y=["Cost", "Time", "Risk"],
                orientation="h",
                marker_color=[COLORS["secondary"], COLORS["info"], COLORS["warning"]],
                text=[f"{sb['weighted_cost']:.4f}", f"{sb['weighted_time']:.4f}", f"{sb['weighted_risk']:.4f}"],
                textposition="outside",
            ))
            apply_chart_theme(fig, title="Weighted Score Contributions", x_title="Score Contribution", height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with breakdown_cols[1]:
            impact = explain_weight_impact(best, weights)
            fig = go.Figure(go.Pie(
                labels=["Cost", "Time", "Risk"],
                values=[impact["cost_pct"], impact["time_pct"], impact["risk_pct"]],
                marker_colors=[COLORS["secondary"], COLORS["info"], COLORS["warning"]],
                hole=0.4,
            ))
            apply_chart_theme(fig, title="Score Composition (%)", height=250)
            st.plotly_chart(fig, use_container_width=True)

        # Full explanation
        st.markdown("---")
        st.markdown('<p class="section-header">💡 Detailed Explanation</p>', unsafe_allow_html=True)
        st.markdown(rec.explanation)

        # Trade-off analysis
        if rec.feasible_count > 1:
            trade_off = generate_trade_off_summary(
                rec.all_routes_scored[rec.all_routes_scored["is_feasible"] == True],
                rec.recommended_route_id,
            )
            st.markdown("---")
            st.markdown(trade_off)

    else:
        # No feasible route
        st.warning("⚠️ No Feasible Route Found")
        st.markdown(rec.explanation)

        relaxations = compute_required_relaxations(rec.constraint_results)
        if relaxations:
            st.markdown('<p class="section-header">🔧 Actionable Concessions to Restore Feasibility</p>', unsafe_allow_html=True)
            rx_rows = []
            for rx in relaxations:
                rx_rows.append({
                    "Closest Route": rx["route_id"],
                    "Constrained Metric": rx["metric"],
                    "Current Route Value": rx["actual_value"],
                    "Allowed Limit": rx["limit"],
                    "Required Increase": f"+{rx['deficit_pct']:.1f}% (+{rx['deficit']:.2f})",
                })
            st.dataframe(pd.DataFrame(rx_rows), use_container_width=True, hide_index=True)

    # CSV download
    st.markdown("---")
    csv_data = rec.all_routes_scored.to_csv(index=False)
    st.download_button(
        "📥 Download Route Analysis (CSV)",
        csv_data,
        f"route_analysis_{selected_sid}.csv",
        "text/csv",
    )


# ─────────────────────────────────────────────────────────
#  MODULE 6: SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────
elif page == "📈 Sensitivity Analysis":
    st.markdown("""
    <div class="app-header">
        <h1>📈 Sensitivity Analysis</h1>
        <p>Explore how weight and constraint changes affect route recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Shipment selection
    selected_sid = st.selectbox(
        "Select Shipment",
        shipments_df["shipment_id"].tolist(),
        format_func=lambda x: f"{x} — {shipments_df[shipments_df['shipment_id']==x].iloc[0]['origin']} → {shipments_df[shipments_df['shipment_id']==x].iloc[0]['destination']}",
        key="sens_shipment",
    )

    shipment = shipments_df[shipments_df["shipment_id"] == selected_sid].iloc[0]
    ship_routes_raw = routes_df[routes_df["shipment_id"] == selected_sid].copy()

    # ── Weight Sensitivity ──
    st.markdown('<p class="section-header">⚖️ Weight Sensitivity</p>', unsafe_allow_html=True)
    st.caption("How do changes in priority weights affect the recommended route?")

    with st.spinner("Computing weight sensitivity analysis..."):
        ws_results = weight_sensitivity_analysis(
            ship_routes_raw,
            shipment["max_budget"],
            shipment["max_transit_days"],
            shipment["max_delay_probability"],
            weight_steps=11,
        )

    if ws_results.empty:
        st.warning("No feasible routes for weight sensitivity analysis.")
    else:
        ws_summary = summarize_weight_sensitivity(ws_results)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            render_kpi(ws_summary["unique_routes"], "Unique Routes Selected")
        with sc2:
            render_kpi(ws_summary["most_frequent"], "Most Stable Route")
        with sc3:
            render_kpi(f"{ws_summary['most_frequent_pct']}%", "Selection Frequency")

        st.markdown("<br>", unsafe_allow_html=True)

        # Route distribution chart
        route_dist = ws_results["recommended_route"].value_counts().reset_index()
        route_dist.columns = ["Route", "Count"]
        fig = px.bar(
            route_dist, x="Route", y="Count",
            color="Route",
            color_discrete_sequence=COLORS["chart_palette"],
        )
        apply_chart_theme(fig, title="Route Selection Frequency Across Weight Combinations", x_title="Route", y_title="Frequency Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap: cost weight vs time weight → recommended route
        st.markdown("**Weight Combination Heatmap:**")

        # Create a pivot for visualization
        pivot_data = ws_results.copy()
        pivot_data["route_code"] = pd.Categorical(pivot_data["recommended_route"]).codes
        pivot = pivot_data.pivot_table(
            index="cost_weight", columns="time_weight",
            values="route_code", aggfunc="first",
        )

        fig = px.imshow(
            pivot.values,
            x=[f"T={t:.1f}" for t in pivot.columns],
            y=[f"C={c:.1f}" for c in pivot.index],
            labels={"x": "Time Weight", "y": "Cost Weight", "color": "Route Code"},
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        apply_chart_theme(fig, title="Recommended Route by Weight Combination (color = route code)", x_title="Time Weight", y_title="Cost Weight")
        st.plotly_chart(fig, use_container_width=True)

        # Explanation
        st.info(
            f"**Interpretation:** The recommended route changes in "
            f"{ws_summary['unique_routes']} different ways as weights vary. "
            f"Route **{ws_summary['most_frequent']}** is the most robust selection, "
            f"preferred in {ws_summary['most_frequent_pct']}% of weight combinations. "
            f"A higher stability percentage indicates a route that performs well "
            f"across different priority profiles."
        )

    # ── Constraint Sensitivity ──
    st.markdown("---")
    st.markdown('<p class="section-header">🔒 Constraint Sensitivity</p>', unsafe_allow_html=True)
    st.caption("How do changes in constraint bounds affect feasibility and recommendations?")

    with st.spinner("Computing constraint sensitivity..."):
        cs_results = constraint_sensitivity(
            ship_routes_raw,
            shipment["max_budget"],
            shipment["max_transit_days"],
            shipment["max_delay_probability"],
            weights,
            steps=10,
        )

    cs_tabs = st.tabs(["Budget", "Transit Time", "Delay Probability"])

    for tab, (dim_name, dim_df) in zip(cs_tabs, cs_results.items()):
        with tab:
            if dim_df.empty:
                st.warning(f"No data for {dim_name} sensitivity.")
                continue

            val_col = [c for c in dim_df.columns if "value" in c][0]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Feasible Route Count", "Optimization Score"),
            )

            fig.add_trace(
                go.Scatter(
                    x=dim_df["multiplier"], y=dim_df["feasible_count"],
                    mode="lines+markers", name="Feasible",
                    marker_color=COLORS["success"],
                ),
                row=1, col=1,
            )

            scores = dim_df["optimization_score"].dropna()
            if not scores.empty:
                fig.add_trace(
                    go.Scatter(
                        x=dim_df.loc[scores.index, "multiplier"],
                        y=scores,
                        mode="lines+markers", name="Opt Score",
                        marker_color=COLORS["secondary"],
                    ),
                    row=1, col=2,
                )

            apply_chart_theme(fig, title=f"{dim_name.title()} Constraint Sensitivity", height=350, showlegend=False)
            fig.update_xaxes(title_text=f"{dim_name.title()} Multiplier", row=1, col=1)
            fig.update_xaxes(title_text=f"{dim_name.title()} Multiplier", row=1, col=2)
            for annotation in fig['layout']['annotations']:
                annotation['font'] = dict(color='#f8fafc', size=13, family='Inter, sans-serif')
            st.plotly_chart(fig, use_container_width=True)

            # Show which route changes
            route_changes = dim_df[["multiplier", val_col, "feasible_count", "recommended_route"]].copy()
            route_changes.columns = ["Multiplier", f"{dim_name.title()} Value", "Feasible Routes", "Recommended Route"]
            st.dataframe(route_changes, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────
#  MODULE 7: EVALUATION & BASELINES
# ─────────────────────────────────────────────────────────
elif page == "📋 Evaluation & Baselines":
    st.markdown("""
    <div class="app-header">
        <h1>📋 Evaluation & Baseline Comparison</h1>
        <p>Comprehensive evaluation of the optimization engine across all shipments</p>
    </div>
    <div class="data-disclaimer">
        ℹ️ <b>Methodological Evaluation:</b> Performance metrics across 35 benchmark shipments comparing Multi-Objective Weighted Optimization against single-objective baselines and random heuristics under strict identical constraints.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running full evaluation across all shipments..."):
        eval_results = run_full_evaluation(shipments_df, routes_df, weights)

    eval_df = evaluation_to_dataframe(eval_results)

    # Summary KPIs
    wo = eval_results["Weighted Optimization"]
    st.markdown('<p class="section-header">Weighted Optimization Summary</p>', unsafe_allow_html=True)

    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    with ec1:
        render_kpi(wo.shipments_evaluated, "Shipments Evaluated")
    with ec2:
        render_kpi(wo.shipments_with_feasible, "With Feasible Routes")
    with ec3:
        render_kpi(wo.shipments_no_feasible, "No Feasible Route")
    with ec4:
        render_kpi(f"{wo.pct_constraints_satisfied}%", "Constraints Satisfied")
    with ec5:
        render_kpi(f"{wo.avg_optimization_score:.4f}" if wo.avg_optimization_score else "N/A", "Avg Opt Score")

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison table
    st.markdown('<p class="section-header">Strategy Comparison</p>', unsafe_allow_html=True)

    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.info(
        "**Note:** All strategies use the same hard constraints. The 'Avg Opt Score' column "
        "shows each strategy's average weighted optimization score (using default weights). "
        "Weighted optimization minimizes this score by design, so it will generally have "
        "the lowest value. This is a decision-support framework, not a proof of global optimality."
    )

    # Comparison charts
    st.markdown("---")
    st.markdown('<p class="section-header">Visual Comparison</p>', unsafe_allow_html=True)

    chart_tabs = st.tabs(["Average Cost", "Average Transit", "Average Delay", "Optimization Score"])

    strategies = eval_df["Strategy"].tolist()
    bar_colors = [COLORS["primary"], COLORS["secondary"], COLORS["info"], COLORS["success"]]

    with chart_tabs[0]:
        fig = px.bar(
            eval_df, x="Strategy", y="Avg Cost ($)",
            color="Strategy",
            color_discrete_sequence=bar_colors,
        )
        apply_chart_theme(fig, title="Average Shipping Cost by Strategy", x_title="Strategy", y_title="Avg Cost ($)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        fig = px.bar(
            eval_df, x="Strategy", y="Avg Transit (days)",
            color="Strategy",
            color_discrete_sequence=bar_colors,
        )
        apply_chart_theme(fig, title="Average Transit Time by Strategy", x_title="Strategy", y_title="Avg Transit (days)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[2]:
        fig = px.bar(
            eval_df, x="Strategy", y="Avg Delay Prob",
            color="Strategy",
            color_discrete_sequence=bar_colors,
        )
        apply_chart_theme(fig, title="Average Delay Probability by Strategy", x_title="Strategy", y_title="Avg Delay Prob", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[3]:
        opt_scores = eval_df[eval_df["Avg Opt Score"].notna()]
        fig = px.bar(
            opt_scores, x="Strategy", y="Avg Opt Score",
            color="Strategy",
            color_discrete_sequence=bar_colors,
        )
        apply_chart_theme(fig, title="Average Optimization Score by Strategy (lower is better)", x_title="Strategy", y_title="Avg Opt Score", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Trade-off radar
    st.markdown("---")
    st.markdown('<p class="section-header">Strategy Trade-off Radar</p>', unsafe_allow_html=True)

    # Normalize metrics for radar comparison
    metrics = ["Avg Cost ($)", "Avg Transit (days)", "Avg Delay Prob"]
    fig = go.Figure()

    for _, row in eval_df.iterrows():
        vals = []
        for m in metrics:
            max_val = eval_df[m].max()
            vals.append(row[m] / max_val if max_val > 0 else 0)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=["Cost", "Transit Time", "Delay Prob", "Cost"],
            name=row["Strategy"],
            fill="toself",
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30, 41, 59, 0.4)",
            radialaxis=dict(
                visible=True,
                range=[0, 1.1],
                gridcolor="rgba(51, 65, 85, 0.5)",
                tickfont=dict(color="#94a3b8"),
                linecolor="rgba(51, 65, 85, 0.6)",
            ),
            angularaxis=dict(
                gridcolor="rgba(51, 65, 85, 0.5)",
                tickfont=dict(color="#cbd5e1"),
                linecolor="rgba(51, 65, 85, 0.6)",
            ),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.75)",
        plot_bgcolor="rgba(30, 41, 59, 0.50)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        title_font=dict(color="#f8fafc", size=14),
        legend=dict(
            font=dict(color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.65)",
            bordercolor="rgba(51, 65, 85, 0.5)",
        ),
        title="Strategy Comparison Radar (lower = better)",
        height=450,
        margin=dict(t=50, b=30, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV download
    st.markdown("---")
    csv_data = eval_df.to_csv(index=False)
    st.download_button(
        "📥 Download Evaluation Results (CSV)",
        csv_data,
        "evaluation_results.csv",
        "text/csv",
    )
