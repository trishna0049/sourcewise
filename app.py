"""SourceWise — Streamlit UI.

All calculations live in the ``sourcewise`` package; this file only handles
inputs, layout and charts. Run with ``streamlit run app.py``.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sourcewise.capacity import ScalingInputs, check_scaling
from sourcewise.formatting import indian_group, inr
from sourcewise.models import DISPLAY_LABELS, OPTIONAL_COLUMNS, SupplierDataError, load_suppliers, \
    validate_suppliers
from sourcewise.moq_risk import (
    DEFAULT_PLANNED_QUANTITY,
    DEFAULT_SCENARIOS,
    DEFAULT_SELLING_PRICE,
    Scenario,
    cheaper_but_riskier,
    low_scenario,
    run_risk_model,
    summarize_risk,
    validate_scenarios,
)
from sourcewise.recommend import (
    DecisionWeights,
    Recommendation,
    WarningThresholds,
    build_recommendation,
    summary_to_excel,
)
from sourcewise.scoring import DEFAULT_WEIGHTS, FACTORS, ScorecardResult, normalize_weights, \
    score_suppliers

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_suppliers.csv"

# Chart colours: categorical slots in fixed order (one per scorecard factor),
# a one-hue blue ramp for ordered demand scenarios (low = lightest), and the
# reserved "serious" status colour for cheaper-but-riskier flags.
FACTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SCENARIO_COLORS = ["#6da7ec", "#2a78d6", "#184f95"]
NEUTRAL_COLOR = "#2a78d6"
FLAG_COLOR = "#ec835a"
STATUS_ICONS = {"Green": "🟢 Green", "Amber": "🟠 Amber", "Red": "🔴 Red"}
CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.set_page_config(page_title="SourceWise — Supplier Evaluation", page_icon="🧵", layout="wide")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    """Load and validate the bundled sample supplier file."""
    return load_suppliers(SAMPLE_PATH)


def show_data_errors(container: st.delta_generator.DeltaGenerator, title: str,
                     error: SupplierDataError) -> None:
    """Render validation errors as one readable block."""
    container.error(f"**{title}**\n\n" + "\n".join(f"- {e}" for e in error.errors))


def sidebar_data_source() -> tuple[pd.DataFrame | None, str, SupplierDataError | None]:
    """Choose sample data or an uploaded CSV. Returns (data, source key, error)."""
    st.sidebar.header("1. Supplier data")
    source = st.sidebar.radio("Data source", ["Sample data (30 suppliers)", "Upload my own CSV"],
                              label_visibility="collapsed")
    st.sidebar.download_button("Download CSV template", SAMPLE_PATH.read_bytes(),
                               file_name="sourcewise_template.csv", mime="text/csv",
                               help="The sample file — use it as a template for your own suppliers.")
    if source.startswith("Sample"):
        return load_sample(), "sample", None
    upload = st.sidebar.file_uploader("Upload supplier CSV", type=["csv"],
                                      help="One row per supplier, same columns as the template.")
    if upload is None:
        st.sidebar.info("No file uploaded yet — showing the sample data.")
        return load_sample(), "sample", None
    try:
        return load_suppliers(io.BytesIO(upload.getvalue())), f"upload-{upload.file_id}", None
    except SupplierDataError as exc:
        return None, "upload-error", exc


def reset_weights() -> None:
    """Callback: put every weight slider back to its default."""
    for key, value in DEFAULT_WEIGHTS.items():
        st.session_state[f"w_{key}"] = int(value)


def sidebar_weights() -> dict[str, float]:
    """Weight sliders; returns raw (un-normalised) weights."""
    st.sidebar.header("3. Scorecard weights")
    st.sidebar.caption("How much each factor matters to you (0 = ignore, 10 = critical). "
                       "Weights are rescaled to add up to 100%.")
    weights: dict[str, float] = {}
    for factor in FACTORS:
        key = f"w_{factor.key}"
        st.session_state.setdefault(key, int(DEFAULT_WEIGHTS[factor.key]))
        weights[factor.key] = st.sidebar.slider(factor.label, 0, 10, key=key, help=factor.help)
    st.sidebar.button("Reset weights", on_click=reset_weights)
    if sum(weights.values()) > 0:
        shares = normalize_weights(weights)
        st.sidebar.caption(" · ".join(f"{f.label} {shares[f.key]:.0%}" for f in FACTORS))
    return weights


def apply_filters(df: pd.DataFrame, container: st.delta_generator.DeltaGenerator) -> pd.DataFrame:
    """State / city / specialization filters, shown only for columns that exist."""
    present = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    container.header("2. Filters")
    if not present:
        container.caption("Add city, state or specialization columns to your CSV to filter by them.")
        return df
    filtered = df
    if "state" in df.columns:
        states = container.multiselect("State", sorted(df["state"].dropna().unique()),
                                       placeholder="All states")
        if states:
            filtered = filtered[filtered["state"].isin(states)]
    if "city" in df.columns:
        cities = container.multiselect("City", sorted(filtered["city"].dropna().unique()),
                                       placeholder="All cities")
        if cities:
            filtered = filtered[filtered["city"].isin(cities)]
    if "specialization" in df.columns:
        specs = container.multiselect("Specialization", sorted(df["specialization"].dropna().unique()),
                                      placeholder="All specializations")
        if specs:
            filtered = filtered[filtered["specialization"].isin(specs)]
    if len(filtered) < len(df):
        container.caption(f"Analysing **{len(filtered)} of {len(df)}** suppliers.")
    return filtered.reset_index(drop=True)


# --------------------------------------------------------------------------
# Table helpers
# --------------------------------------------------------------------------

SUPPLIER_COLUMN_CONFIG = {
    "supplier_name": st.column_config.TextColumn("Supplier", pinned=True, required=True),
    "city": st.column_config.TextColumn("City"),
    "state": st.column_config.TextColumn("State"),
    "specialization": st.column_config.TextColumn("Specialization"),
    "unit_cost": st.column_config.NumberColumn("Unit cost (₹)", format="₹%d", min_value=0,
                                               help="Price per piece."),
    "moq_per_style": st.column_config.NumberColumn("MOQ / style", min_value=1, step=1, format="%d",
                                                   help="Minimum units per style."),
    "sample_quality": st.column_config.NumberColumn("Quality", min_value=1, max_value=5, step=1,
                                                    help="Sample quality, 1–5."),
    "capability_score": st.column_config.NumberColumn("Capability", min_value=1, max_value=5, step=1,
                                                      help="Stitching / finishing / silhouette, 1–5."),
    "meets_required_capability": st.column_config.CheckboxColumn(
        "Meets capability", help="Hard requirement: can they make your garment to spec?"),
    "lead_time_days": st.column_config.NumberColumn("Lead time (d)", min_value=0, step=1, format="%d",
                                                    help="Days to first production."),
    "reorder_time_days": st.column_config.NumberColumn("Reorder (d)", min_value=0, step=1, format="%d",
                                                       help="Days to restock a repeat order."),
    "monthly_capacity": st.column_config.NumberColumn("Capacity / month", min_value=0, step=100,
                                                      format="%d"),
    "reliability_score": st.column_config.NumberColumn("Reliability", min_value=1, max_value=5, step=1,
                                                       help="Communication & meeting deadlines, 1–5."),
    "costing_transparency": st.column_config.NumberColumn("Transparency", min_value=1, max_value=5,
                                                          step=1, help="Openness of costing, 1–5."),
    "advance_payment_pct": st.column_config.NumberColumn("Advance (%)", min_value=0, max_value=100,
                                                         step=5, format="%d%%"),
}


def descriptive_cols(df: pd.DataFrame) -> list[str]:
    """Optional descriptive columns present in ``df``."""
    return [c for c in OPTIONAL_COLUMNS if c in df.columns]


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def _raw_text(column: str, value: float) -> str:
    """Raw factor value formatted for tooltips."""
    if column == "unit_cost":
        return inr(value)
    if column == "advance_payment_pct":
        return f"{value:g}%"
    if column in ("lead_time_days",):
        return f"{value:g} days"
    if column == "moq_per_style":
        return f"{indian_group(value)} units"
    return f"{value:g} / 5"


def _base_layout(fig: go.Figure, height: int, x_title: str, legend_rows: int = 1) -> go.Figure:
    """Shared layout: legend in a row above the plot, recessive grid, top-to-bottom bars."""
    top = 20 + 24 * legend_rows
    fig.update_layout(
        height=height + top, margin=dict(l=10, r=30, t=top, b=10),
        legend=dict(orientation="h", yanchor="top", y=1.0, yref="container", xanchor="left", x=0,
                    title=None, traceorder="normal"),
        hoverlabel=dict(align="left"), bargap=0.3,
    )
    fig.update_xaxes(title=x_title, showgrid=True, gridwidth=1, zeroline=True, zerolinewidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, title=None)
    return fig


def scorecard_chart(ranked: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bars: points each factor contributes to the score."""
    fig = go.Figure()
    for color, factor in zip(FACTOR_COLORS, FACTORS):
        raw = [_raw_text(factor.column, v) for v in ranked[factor.column]]
        fig.add_bar(
            y=ranked["supplier_name"], x=ranked[f"pts_{factor.key}"], name=factor.label,
            orientation="h", marker=dict(color=color, cornerradius=0),
            customdata=list(zip(raw, ranked["score"])),
            hovertemplate=(f"<b>%{{y}}</b><br>{factor.label}: %{{x:.1f}} pts"
                           "<br>Raw value: %{customdata[0]}<br>Total score: %{customdata[1]:.1f}"
                           "<extra></extra>"),
        )
    fig.add_scatter(x=ranked["score"], y=ranked["supplier_name"], mode="text",
                    text=[f" {s:.0f}" for s in ranked["score"]], textposition="middle right",
                    showlegend=False, hoverinfo="skip", cliponaxis=False)
    fig.update_layout(barmode="stack")
    fig = _base_layout(fig, 60 + 30 * len(ranked), "Score (out of 100)", legend_rows=2)
    fig.update_xaxes(range=[0, 105])
    return fig


def profit_chart(results: pd.DataFrame, scenarios: list[Scenario], order: list[str]) -> go.Figure:
    """Grouped horizontal bars: profit per supplier under each demand scenario."""
    fig = go.Figure()
    by_demand = sorted(scenarios, key=lambda s: s.demand)
    for color, sc in zip(SCENARIO_COLORS, by_demand):
        part = results[results["scenario"] == sc.name].set_index("supplier_name").loc[order]
        custom = [(inr(p), indian_group(sold), indian_group(bought), inr(car))
                  for p, sold, bought, car in zip(part["profit"], part["units_sold"],
                                                   part["units_bought"], part["capital_at_risk"])]
        fig.add_bar(
            y=order, x=part["profit"], name=f"{sc.name} ({indian_group(sc.demand)} units)",
            orientation="h", marker=dict(color=color, cornerradius=4), customdata=custom,
            hovertemplate=("<b>%{y}</b><br>" + sc.name + " demand<br>Profit: %{customdata[0]}"
                           "<br>Sold %{customdata[1]} of %{customdata[2]} bought"
                           "<br>Capital at risk: %{customdata[3]}<extra></extra>"),
        )
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig = _base_layout(fig, 70 + 75 * len(order), "Profit per style (₹)", legend_rows=2)
    fig.update_xaxes(tickprefix="₹", separatethousands=True)
    return fig


def capital_at_risk_chart(results: pd.DataFrame, scenario: str,
                          flagged: set[str]) -> go.Figure:
    """Capital tied up in unsold stock per supplier, cheapest supplier first."""
    data = (results[results["scenario"] == scenario].sort_values(["unit_cost", "supplier_name"])
            .reset_index(drop=True))
    labels = [f"{n}  ·  {inr(c)}  ·  MOQ {indian_group(m)}"
              for n, c, m in zip(data["supplier_name"], data["unit_cost"], data["moq_per_style"])]
    is_flag = data["supplier_name"].isin(flagged)
    fig = go.Figure()
    groups = [(~is_flag, "Other suppliers", NEUTRAL_COLOR), (is_flag, "⚠ Cheaper but riskier", FLAG_COLOR)]
    for mask, name, color in groups:
        if not mask.any():
            continue
        part = data[mask]
        fig.add_bar(
            y=[labels[i] for i in part.index], x=part["capital_at_risk"], name=name, orientation="h",
            marker=dict(color=color, cornerradius=4),
            text=[inr(v) for v in part["capital_at_risk"]], textposition="outside", cliponaxis=False,
            customdata=[(inr(v), indian_group(u)) for v, u in zip(part["capital_at_risk"],
                                                                 part["unsold_units"])],
            hovertemplate="<b>%{y}</b><br>Capital at risk: %{customdata[0]}"
                          "<br>Unsold units: %{customdata[1]}<extra></extra>",
            showlegend=bool(is_flag.any()),
        )
    fig = _base_layout(fig, 70 + 42 * len(data), "Capital at risk (₹)")
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    fig.update_xaxes(tickprefix="₹", separatethousands=True,
                     range=[0, max(float(data["capital_at_risk"].max()), 1.0) * 1.25])
    return fig


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

def suppliers_tab(filtered: pd.DataFrame, all_rows: int) -> None:
    """Overview metrics and the (filtered) supplier table."""
    eligible = int(filtered["meets_required_capability"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suppliers", len(filtered), help=f"{all_rows} in the file before filters.")
    c2.metric("Meet required capability", f"{eligible} of {len(filtered)}",
               help="Suppliers that fail the hard capability requirement are never ranked.")
    c3.metric("Median unit cost", inr(filtered["unit_cost"].median()))
    c4.metric("Lowest MOQ", f"{indian_group(filtered['moq_per_style'].min())} units")
    st.dataframe(filtered, hide_index=True, column_config=SUPPLIER_COLUMN_CONFIG,
                 height=min(38 + 35 * len(filtered), 560))


def scorecard_tab(result: ScorecardResult) -> None:
    """Ranked table, contribution chart and excluded suppliers."""
    with st.expander("How the score works"):
        st.markdown(
            "1. **Hard filter** — suppliers that cannot make your garment to spec are removed first.\n"
            "2. **Normalise** — each factor is rescaled to 0–1 across the remaining suppliers: "
            "best = 1, worst = 0. For cost, MOQ, lead time and advance payment *lower is better*, "
            "so the scale is flipped. If every supplier has the same value, all get 1.\n"
            "3. **Weight** — `score = Σ (normalised factor × weight) × 100`, so 100 means best "
            "on every factor you care about."
        )
        st.latex(r"\text{norm}(x) = \frac{x - \min}{\max - \min}\quad"
                 r"\text{or}\quad \frac{\max - x}{\max - \min}\ \text{(lower is better)}")
    ranked = result.ranked
    if ranked.empty:
        st.warning("No supplier meets the required capability, so there is nothing to rank.")
    else:
        top = ranked.iloc[0]
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.metric("Top supplier", top["supplier_name"])
        c2.metric("Top score", f"{top['score']:.1f} / 100")
        c3.metric("Ranked / excluded", f"{len(ranked)} / {len(result.excluded)}")

        st.subheader("Ranking")
        cols = ["rank", "supplier_name", *descriptive_cols(ranked), "score", "unit_cost",
                "moq_per_style", "sample_quality", "capability_score", "lead_time_days",
                "reliability_score", "costing_transparency", "advance_payment_pct"]
        config = {**SUPPLIER_COLUMN_CONFIG,
                  "rank": st.column_config.NumberColumn("Rank", format="%d", pinned=True),
                  "score": st.column_config.ProgressColumn(
                      "Score", min_value=0, max_value=100, format="%.1f",
                      help="Weighted score out of 100.")}
        st.dataframe(ranked[cols], hide_index=True, column_config=config,
                     height=min(38 + 35 * len(ranked), 460))

        st.subheader("What drives each score")
        show_all = st.toggle(f"Show all {len(ranked)} suppliers", value=False,
                             disabled=len(ranked) <= 10)
        shown = ranked if show_all else ranked.head(10)
        st.caption("Each bar segment is the points a factor adds to the score (out of 100). "
                   "A long segment means the supplier is strong on a factor you weighted highly.")
        st.plotly_chart(scorecard_chart(shown), config=CHART_CONFIG)

    st.subheader(f"Excluded by hard filter ({len(result.excluded)})")
    if result.excluded.empty:
        st.success("Every supplier meets the required capability.")
    else:
        cols = ["supplier_name", *descriptive_cols(result.excluded), "unit_cost", "moq_per_style",
                "capability_score", "exclusion_reason"]
        st.dataframe(result.excluded[cols], hide_index=True, column_config={
            **SUPPLIER_COLUMN_CONFIG, "exclusion_reason": st.column_config.TextColumn("Reason")})


def reset_comparison(default: list[str]) -> None:
    """Callback: compare the scorecard's top suppliers again."""
    st.session_state["compare"] = default


def risk_inputs(ranked: pd.DataFrame) -> tuple[float, int, list[Scenario], list[str], str | None]:
    """Inputs for the risk model. Returns (price, quantity, scenarios, suppliers, error)."""
    c1, c2 = st.columns(2)
    price = c1.number_input("Selling price per unit (₹)", min_value=1.0, value=DEFAULT_SELLING_PRICE,
                            step=10.0, format="%.0f",
                            help="What a customer pays you for one piece (after discounts).")
    quantity = int(c2.number_input(
        "Planned order quantity per style (units)", min_value=1, value=DEFAULT_PLANNED_QUANTITY,
        step=10, help="What you would order if there were no MOQ. If a supplier's MOQ is higher, "
                      "you are forced to buy the MOQ."))

    st.markdown("**Demand scenarios** — how many units of this style could you sell?")
    default_table = pd.DataFrame({
        "Scenario": [s.name for s in DEFAULT_SCENARIOS],
        "Demand (units)": [s.demand for s in DEFAULT_SCENARIOS],
        "Probability (%)": [s.probability for s in DEFAULT_SCENARIOS],
    })
    edited = st.data_editor(
        default_table, key="scenarios", hide_index=True, num_rows="fixed", width="content",
        column_config={
            "Scenario": st.column_config.TextColumn(disabled=True),
            "Demand (units)": st.column_config.NumberColumn(min_value=0, step=5, format="%d",
                                                            required=True),
            "Probability (%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=5,
                                                             format="%d%%", required=True),
        })
    error = None
    scenarios: list[Scenario] = []
    if edited[["Demand (units)", "Probability (%)"]].isna().any().any():
        error = "Fill in demand and probability for every scenario."
    else:
        scenarios = [Scenario(str(r["Scenario"]), int(r["Demand (units)"]), float(r["Probability (%)"]))
                     for _, r in edited.iterrows()]
        try:
            validate_scenarios(scenarios)
        except ValueError as exc:
            error = str(exc)

    options = ranked["supplier_name"].tolist()
    default = options[:5]
    current = [n for n in st.session_state.get("compare", default) if n in options]
    st.session_state["compare"] = current or default
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    chosen = c1.multiselect("Suppliers to compare", options, key="compare",
                            help="Defaults to the top 5 from the scorecard. Only suppliers that "
                                 "meet the capability requirement can be compared.")
    c2.button("Reset to top 5", on_click=reset_comparison, args=(default,))
    return float(price), quantity, scenarios, chosen, error


def risk_tab(ranked: pd.DataFrame) -> dict[str, object] | None:
    """MOQ vs unit-cost risk model. Returns the model outputs, or None if not runnable."""
    st.markdown(
        "A cheaper supplier with a **high MOQ** can force you to buy more than you can sell. "
        "This compares profit and **capital at risk** (money stuck in unsold stock) for each "
        "supplier under low, expected and high demand."
    )
    with st.expander("How the risk model works"):
        st.markdown(
            "For each supplier and scenario:\n\n"
            "- `units bought = max(MOQ, planned quantity)`\n"
            "- `units sold = min(units bought, demand)`\n"
            "- `profit = units sold × selling price − units bought × unit cost`\n"
            "- `capital at risk = unsold units × unit cost`\n\n"
            "**Expected profit** weights each scenario by its probability. **Break-even units** is "
            "how many pieces you must sell to recover the order cost. The model does not include "
            "reorders, so in the high scenario a small order simply sells out."
        )
    if ranked.empty:
        st.warning("No eligible suppliers to compare.")
        return None
    price, quantity, scenarios, chosen, error = risk_inputs(ranked)
    if error:
        st.error(error)
        return None
    if not chosen:
        st.info("Pick at least one supplier to compare.")
        return None

    suppliers = ranked.set_index("supplier_name").loc[chosen].reset_index()
    results = run_risk_model(suppliers, price, quantity, scenarios)
    summary = summarize_risk(results, suppliers, scenarios, price)
    pairs = cheaper_but_riskier(summary)
    low = low_scenario(scenarios)

    st.subheader("Results per style")
    table = pd.DataFrame({
        "Supplier": summary["supplier_name"],
        "Unit cost": summary["unit_cost"].map(inr),
        "MOQ": summary["moq_per_style"],
        "Units bought": summary["units_bought"],
        "Order value": summary["order_value"].map(inr),
        "Advance": summary["advance_amount"].map(inr),
        "Break-even units": summary["breakeven_units"],
        "Expected profit": summary["expected_profit"].map(inr),
        f"Profit if demand {low.name.lower()}": summary["low_profit"].map(inr),
        f"Capital at risk if {low.name.lower()}": summary["low_capital_at_risk"].map(inr),
        "Flag": ["⚠ Cheaper but riskier" if n in set(pairs["cheaper_supplier"]) else ""
                 for n in summary["supplier_name"]],
    })
    st.dataframe(table, hide_index=True, column_config={
        "Units bought": st.column_config.NumberColumn(help="max(MOQ, planned quantity)"),
        "Break-even units": st.column_config.NumberColumn(
            help="Units you must sell to recover the order cost."),
        "Expected profit": st.column_config.TextColumn(help="Probability-weighted profit."),
    })

    if pairs.empty:
        st.success(f"No cheaper supplier carries more capital at risk than a dearer one if demand "
                   f"is {low.name.lower()}.")
    else:
        st.markdown(f"**Cheaper isn't always safer** — if demand is {low.name.lower()} "
                    f"({indian_group(low.demand)} units):")
        for _, p in pairs.head(4).iterrows():
            st.warning(
                f"**{p['cheaper_supplier']}** is {inr(p['unit_cost_saving'])}/unit cheaper than "
                f"**{p['dearer_supplier']}**, but its MOQ of {indian_group(p['cheaper_moq'])} "
                f"(vs {indian_group(p['dearer_moq'])}) puts **{inr(p['extra_capital_at_risk'])} more** "
                "capital at risk.", icon="⚠️")
        if len(pairs) > 4:
            st.caption(f"…and {len(pairs) - 4} more pair(s).")

    left, right = st.columns(2)
    with left:
        st.markdown("**Profit by demand scenario**")
        st.plotly_chart(profit_chart(results, scenarios, chosen), config=CHART_CONFIG)
    with right:
        names = [s.name for s in sorted(scenarios, key=lambda s: s.demand)]
        picked = st.segmented_control("Capital at risk if demand is", names, default=low.name,
                                      key="car_scenario") or low.name
        flagged = set(pairs["cheaper_supplier"]) if picked == low.name else set()
        st.plotly_chart(capital_at_risk_chart(results, picked, flagged), config=CHART_CONFIG)
        st.caption("Sorted cheapest first. Orange bars are cheaper suppliers that leave more money "
                   "stuck in unsold stock than a dearer option.")

    with st.expander("Full scenario detail"):
        detail = results[["supplier_name", "scenario", "demand", "units_bought", "units_sold",
                          "unsold_units", "revenue", "cost", "profit", "capital_at_risk"]].copy()
        for col in ("revenue", "cost", "profit", "capital_at_risk"):
            detail[col] = detail[col].map(inr)
        st.dataframe(detail.rename(columns=lambda c: c.replace("_", " ").capitalize()),
                     hide_index=True)

    return {"price": price, "quantity": quantity, "scenarios": scenarios, "results": results,
            "summary": summary, "low": low}


def scaling_tab(ranked: pd.DataFrame, compared: list[str]) -> tuple[pd.DataFrame, ScalingInputs] | None:
    """Capacity and reorder feasibility for every eligible supplier."""
    st.markdown("If a style takes off, can the supplier keep up? Checks **monthly capacity** "
                "against your grown volume and **reorder time** against how long you can wait.")
    c1, c2, c3, c4 = st.columns(4)
    current = int(c1.number_input("Current monthly volume (units)", min_value=0, value=800, step=100,
                                  help="Units you order per month today, across all styles."))
    growth = c2.slider("Demand growth multiplier", 1.0, 10.0, 4.0, 0.5, format="%.1f×",
                       help="E.g. 4× if you expect to sell four times as much.")
    deadline = int(c3.number_input("Reorder needed within (days)", min_value=0, value=20, step=1,
                                   help="Longest you can wait for a restock before losing sales."))
    share = c4.slider("Usable share of their capacity", 10, 100, 100, 5, format="%d%%",
                      help="Suppliers serve other brands too. 50% means you can count on half "
                           "their monthly capacity.")
    inputs = ScalingInputs(current, growth, deadline, share / 100)
    if ranked.empty:
        st.warning("No eligible suppliers to check.")
        return None
    scaling = check_scaling(ranked, inputs)

    counts = scaling["status"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volume needed", f"{indian_group(inputs.required_monthly_units)} / month")
    m2.metric("🟢 Green", int(counts.get("Green", 0)), help="Capacity and reorder time both OK.")
    m3.metric("🟠 Amber", int(counts.get("Amber", 0)), help="One of the two checks fails.")
    m4.metric("🔴 Red", int(counts.get("Red", 0)), help="Both checks fail.")

    only_compared = st.toggle("Only suppliers compared in MOQ Risk", value=False,
                              disabled=not compared)
    shown = scaling[scaling["supplier_name"].isin(compared)] if only_compared else scaling
    table = pd.DataFrame({
        "Status": shown["status"].map(STATUS_ICONS),
        "Supplier": shown["supplier_name"],
        "Compared": shown["supplier_name"].isin(compared),
        "Capacity / month": shown["usable_capacity"].round(0).astype(int),
        "Utilisation": shown["capacity_utilisation_pct"].clip(upper=999),
        "Reorder (days)": shown["reorder_time_days"],
        "Slack (days)": shown["reorder_slack_days"],
        "Explanation": shown["explanation"],
    })
    st.dataframe(table, hide_index=True, height=38 + 35 * len(table), column_config={
        "Compared": st.column_config.CheckboxColumn(help="Selected in the MOQ Risk tab."),
        "Capacity / month": st.column_config.NumberColumn(help="Usable capacity after your share."),
        "Utilisation": st.column_config.NumberColumn(
            format="%.0f%%", help="Your grown volume as a share of usable capacity. Over 100% = short."),
        "Slack (days)": st.column_config.NumberColumn(
            help="Days to spare against your reorder deadline. Negative = late."),
        "Explanation": st.column_config.TextColumn(width="large", help="Why the supplier got this status."),
    })
    return scaling, inputs


def recommendation_tab(scorecard: ScorecardResult, risk: dict[str, object] | None,
                       scaling_out: tuple[pd.DataFrame, ScalingInputs] | None,
                       raw_weights: dict[str, float]) -> None:
    """Final recommendation, trade-off, warnings, runner-up and export."""
    if risk is None or scaling_out is None:
        st.info("Complete the **MOQ Risk** tab (valid scenarios and at least one supplier) to get a "
                "recommendation.")
        return
    scaling, scaling_inputs = scaling_out

    with st.expander("Decision settings", icon="⚙️"):
        c1, c2, c3 = st.columns(3)
        w_score = c1.slider("Weight on scorecard vs. risk model", 0, 100, 60, 5, format="%d%%",
                            help="The rest goes to the risk model (expected profit and capital at "
                                 "risk if demand is low).")
        amber = c2.number_input("Penalty for Amber scaling (points)", 0, 100, 10)
        red = c3.number_input("Penalty for Red scaling (points)", 0, 100, 25)
        c4, c5, _ = st.columns(3)
        max_lead = int(c4.number_input("Warn if first-production lead time exceeds (days)", 0, 365, 30))
        high_adv = float(c5.number_input("Warn if advance payment is at least (%)", 0, 100, 50))
    weights = DecisionWeights(w_score, 100 - w_score, float(amber), float(red))
    thresholds = WarningThresholds(max_lead, high_adv)
    low: Scenario = risk["low"]  # type: ignore[assignment]

    try:
        rec = build_recommendation(scorecard, risk["summary"], scaling, weights, thresholds,
                                   planned_quantity=risk["quantity"], low_demand=low.demand)
    except ValueError as exc:
        st.error(str(exc))
        return
    best = rec.decision_table.iloc[0]

    st.success(f"### Recommended: {rec.recommended}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Decision score", f"{rec.decision_score:.1f}",
              help="Blend of scorecard and risk model, minus any scaling penalty.")
    m2.metric("Scorecard", f"{rec.scorecard_score:.1f} / 100", f"Rank {rec.scorecard_rank}",
              delta_color="off", delta_arrow="off")
    m3.metric("Expected profit / style", inr(best["expected_profit"]))
    m4.metric(f"At risk if demand {low.name.lower()}", inr(best["low_capital_at_risk"]))
    m5.metric("Scaling", STATUS_ICONS.get(str(best["status"]), "—"))

    st.markdown("#### The key trade-off")
    st.info(rec.trade_off)

    st.markdown("#### Watch out for")
    if rec.warnings:
        for w in rec.warnings:
            st.warning(w, icon="⚠️")
    else:
        st.success("No major warnings on capacity, lead time, advance payment or downside risk.",
                   icon="✅")

    st.markdown("#### Runner-up")
    st.write(rec.runner_up_reason or "Only one supplier was compared, so there is no runner-up.")

    st.markdown("#### Full decision ranking")
    table = rec.decision_table
    st.dataframe(pd.DataFrame({
        "Rank": table["decision_rank"],
        "Supplier": table["supplier_name"],
        "Decision score": table["decision_score"].round(1),
        "Scorecard": table["score"].round(1),
        "Risk score": table["risk_score"].round(1),
        "Scaling": table["status"].map(STATUS_ICONS),
        "Penalty": table["scaling_penalty"],
        "Expected profit": table["expected_profit"].map(inr),
        f"At risk if {low.name.lower()}": table["low_capital_at_risk"].map(inr),
    }), hide_index=True, column_config={
        "Risk score": st.column_config.NumberColumn(
            help="0–100: half expected profit, half low capital at risk, scaled across compared "
                 "suppliers."),
    })
    with st.expander("How the recommendation is decided"):
        st.markdown(
            f"- **Risk score** = 100 × (½ × normalised expected profit + ½ × normalised low-demand "
            f"capital at risk, lower is better)\n"
            f"- **Decision score** = {w_score}% × scorecard + {100 - w_score}% × risk score − "
            f"scaling penalty ({amber} for Amber, {red} for Red)\n"
            "- Suppliers that fail the capability requirement are never recommended. Ties go to the "
            "better scorecard rank."
        )

    st.download_button(
        "⬇️ Download summary (Excel)", type="primary",
        data=export_summary(rec, scorecard, risk, scaling, scaling_inputs, raw_weights, weights),
        file_name=f"sourcewise_summary_{date.today():%Y-%m-%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def export_summary(rec: Recommendation, scorecard: ScorecardResult, risk: dict[str, object],
                   scaling: pd.DataFrame, scaling_inputs: ScalingInputs,
                   raw_weights: dict[str, float], weights: DecisionWeights) -> bytes:
    """Excel bytes for the download button, including the assumptions used."""
    shares = normalize_weights(raw_weights)
    scenarios: list[Scenario] = risk["scenarios"]  # type: ignore[assignment]
    assumptions: dict[str, object] = {
        "Selling price (₹)": risk["price"],
        "Planned quantity per style": risk["quantity"],
        **{f"Scenario {s.name}": f"{s.demand} units @ {s.probability:g}%" for s in scenarios},
        **{f"Weight: {f.label}": f"{shares[f.key]:.0%}" for f in FACTORS},
        "Current monthly volume": scaling_inputs.current_monthly_units,
        "Growth multiplier": scaling_inputs.growth_multiplier,
        "Reorder deadline (days)": scaling_inputs.reorder_deadline_days,
        "Usable capacity share": f"{scaling_inputs.max_capacity_share:.0%}",
        "Decision weights (scorecard / risk)": f"{weights.scorecard:g}% / {weights.risk:g}%",
    }
    return summary_to_excel(rec, scorecard, risk["results"], risk["summary"], scaling, assumptions)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def main() -> None:
    """Lay out the page and wire the analyses together."""
    st.title("🧵 SourceWise")
    st.caption("Supplier evaluation & sourcing decisions for apparel brands — compare manufacturers "
               "on quality, cost, MOQ, lead time and reliability, and see what a high MOQ really "
               "costs you.")
    alerts = st.container()

    data, source_key, load_error = sidebar_data_source()
    filter_box = st.sidebar.container()
    raw_weights = sidebar_weights()

    if load_error is not None:
        show_data_errors(alerts, "Your file could not be used — please fix these problems and "
                                 "upload again:", load_error)
        st.stop()
    assert data is not None

    tabs = st.tabs(["📋 Suppliers", "🏆 Scorecard", "📦 MOQ Risk", "📈 Scaling", "✅ Recommendation"])

    with tabs[0]:
        overview = st.container()
        with st.expander("✏️ Edit supplier data", expanded=False):
            st.caption("Change any cell, or add / delete rows at the bottom. Edits flow through every "
                       "tab (they are not saved to the file). Row numbers in error messages count the "
                       "header as row 1, like a spreadsheet.")
            edited = st.data_editor(data, key=f"editor-{source_key}", num_rows="dynamic",
                                    hide_index=True, column_config=SUPPLIER_COLUMN_CONFIG)
    try:
        suppliers = validate_suppliers(edited)
    except SupplierDataError as exc:
        show_data_errors(alerts, "Your edits contain problems — fix them in the Suppliers tab:", exc)
        st.stop()

    filtered = apply_filters(suppliers, filter_box)
    if filtered.empty:
        alerts.warning("No suppliers match the current filters. Clear a filter in the sidebar.")
        st.stop()
    if sum(raw_weights.values()) == 0:
        alerts.error("Set at least one scorecard weight above zero.")
        st.stop()

    with overview:
        suppliers_tab(filtered, len(suppliers))

    scorecard = score_suppliers(filtered, raw_weights)
    with tabs[1]:
        scorecard_tab(scorecard)
    with tabs[2]:
        risk = risk_tab(scorecard.ranked)
    compared = risk["summary"]["supplier_name"].tolist() if risk else []
    with tabs[3]:
        scaling_out = scaling_tab(scorecard.ranked, compared)
    with tabs[4]:
        recommendation_tab(scorecard, risk, scaling_out, raw_weights)

    st.sidebar.divider()
    st.sidebar.caption("Sample data is synthetic, modelled on Indian apparel manufacturing clusters.")


main()
