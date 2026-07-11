"""
Climate Intelligence Platform — Future Forecast
Policy Target Tracker · Scenario Comparison Summary · All-scenario overlay
"""
import sys, os
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
_APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _SRC)
sys.path.insert(0, _APP)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from features import create_features, get_feature_list
from scenarios import generate_scenario, compare_scenarios, SCENARIOS
from models import train_final_model
from explainability import build_policy_tracker
from styles import DS, SERIES_COLORS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout, add_thresholds

st.set_page_config(page_title="Future Forecast | Climate Platform", page_icon="🔭", layout="wide")
apply_global_css()

# ── Data & model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/climate_data.csv")
    return create_features(df)

@st.cache_resource
def load_model(_df):
    X, y = _df[get_feature_list()], _df["temperature_anomaly"]
    path = "models/xgboost_model.pkl"
    return joblib.load(path) if os.path.exists(path) else train_final_model(X, y, "XGBoost")

@st.cache_data
def get_all_scenarios(_df, end_year):
    model = load_model(_df)
    return compare_scenarios(_df, end_year, get_feature_list(), model)

df = load_data()
model = load_model(df)
features = get_feature_list()
last_hist_year = int(df["year"].max())
FORECAST_END = 2100

all_sc = get_all_scenarios(df, FORECAST_END)

# Baseline: Current Policies
baseline_name = "🟡 Current Policies (+0.5%/yr)"
baseline_df = all_sc[all_sc["scenario"] == baseline_name].sort_values("year")

# Projected temps at milestones
def proj_temp(year):
    row = baseline_df[baseline_df["year"] == year]
    return row["temperature_anomaly"].values[0] if not row.empty else float("nan")

t2030 = proj_temp(2030)
t2050 = proj_temp(2050)
t2100 = proj_temp(2100)
current_temp = df["temperature_anomaly"].iloc[-1]

# ── Header & KPIs ─────────────────────────────────────────────────────────────
page_header("Future Forecast", f"Baseline projection to {FORECAST_END} under Current Policies, with all-scenario comparison")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Current Anomaly", f"{current_temp:+.2f} °C", f"Year {last_hist_year}", "up"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Projected 2030", f"{t2030:+.2f} °C" if not np.isnan(t2030) else "N/A", "Current Policies baseline", "up"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Projected 2050", f"{t2050:+.2f} °C" if not np.isnan(t2050) else "N/A", "Current Policies baseline", "up"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Projected 2100", f"{t2100:+.2f} °C" if not np.isnan(t2100) else "N/A", "Current Policies baseline", "up"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

# ── Baseline forecast chart ───────────────────────────────────────────────────
section_label("Baseline Projection — Current Policies")

fig_base = go.Figure()

# Historical
fig_base.add_trace(go.Scatter(
    x=df["year"], y=df["temperature_anomaly"],
    name="Historical", mode="lines",
    line=dict(color=DS["cyan"], width=2),
))

# Confidence band (±0.2°C)
fig_base.add_trace(go.Scatter(
    x=list(baseline_df["year"]) + list(baseline_df["year"])[::-1],
    y=list(baseline_df["temperature_anomaly"] + 0.2) + list(baseline_df["temperature_anomaly"] - 0.2)[::-1],
    fill="toself", fillcolor="rgba(245,158,11,0.10)",
    line=dict(color="rgba(0,0,0,0)"),
    name="\u00b10.2\u00b0C uncertainty", showlegend=True,
    hoverinfo="skip",
))

# Projected line
fig_base.add_trace(go.Scatter(
    x=baseline_df["year"], y=baseline_df["temperature_anomaly"],
    name="Projected (baseline)", mode="lines",
    line=dict(color=DS["amber"], width=2.5, dash="dash"),
    hovertemplate="Year: %{x}<br>Projected: %{y:.3f} °C<extra></extra>",
))

# Forecast start
fig_base.add_vline(
    x=last_hist_year, line=dict(color="rgba(255,255,255,0.2)", width=1.2, dash="dot"),
    annotation=dict(text="Forecast start", font=dict(size=10, color="#94a3b8"), xanchor="left"),
)

add_thresholds(fig_base)

fig_base.update_layout(**chart_layout(
    title=dict(text="Global Temperature Anomaly — Historical & Projected (Current Policies)"),
    height=420,
    yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.08)"),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig_base, width="stretch")

# ── All-scenario overlay ──────────────────────────────────────────────────────
section_label("All-Scenario Comparison")

fig_all = go.Figure()
fig_all.add_trace(go.Scatter(
    x=df["year"], y=df["temperature_anomaly"],
    name="Historical", mode="lines",
    line=dict(color="rgba(241,245,249,0.5)", width=1.8),
))

sc_colors = [s["color"] for s in SCENARIOS.values()]
for i, (sc_name, sc_cfg) in enumerate(SCENARIOS.items()):
    sc_data = all_sc[all_sc["scenario"] == sc_name].sort_values("year")
    if sc_data.empty:
        continue
    # Endpoint label
    last_row = sc_data.iloc[-1]
    short_name = sc_name.split("(")[0].strip().lstrip("🔴🟡🟢🔵 ")
    fig_all.add_trace(go.Scatter(
        x=sc_data["year"], y=sc_data["temperature_anomaly"],
        name=short_name, mode="lines",
        line=dict(color=sc_cfg["color"], width=2.2),
        hovertemplate=f"<b>{short_name}</b><br>Year: %{{x}}<br>Temp: %{{y:.3f}} °C<extra></extra>",
    ))
    fig_all.add_annotation(
        x=last_row["year"], y=last_row["temperature_anomaly"],
        text=f"{last_row['temperature_anomaly']:+.2f}°C",
        font=dict(size=9, color=sc_cfg["color"]),
        showarrow=False, xanchor="left", xshift=4,
        bgcolor="rgba(10,15,26,0.8)",
    )

add_thresholds(fig_all)
fig_all.add_vline(
    x=last_hist_year, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
)

fig_all.update_layout(**chart_layout(
    title=dict(text="All Emission Scenarios — Projected Temperature to 2100"),
    height=460,
    yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig_all, width="stretch")

# ── Policy Target Tracker ─────────────────────────────────────────────────────
section_label("Policy Target Tracker")

tracker_df = build_policy_tracker(all_sc, thresholds=(1.5, 2.0))

# Style: color cells
def color_crossing(val):
    if val == "Does not cross":
        return "color: #10b981; font-weight:600"
    try:
        yr = int(val)
        if yr <= 2050:
            return "color: #f43f5e; font-weight:600"
        return "color: #f59e0b; font-weight:600"
    except:
        return ""

tracker_disp = tracker_df.copy()
tracker_disp["Scenario"] = tracker_disp["Scenario"].str.lstrip("🔴🟡🟢🔵 ").str.split("(").str[0].str.strip()

st.dataframe(
    tracker_disp.style.applymap(color_crossing, subset=[c for c in tracker_disp.columns if "°C" in c]),
    width="stretch", hide_index=True,
)

# Insight cards
c1, c2 = st.columns(2)
with c1:
    # Find earliest crossing of 1.5°C
    col_15 = [c for c in tracker_df.columns if "1.5" in c]
    if col_15:
        crossings = tracker_df[col_15[0]].replace("Does not cross", None).dropna()
        if not crossings.empty:
            earliest_yr = crossings.astype(int).min()
            earliest_sc = tracker_df.loc[tracker_df[col_15[0]] == str(earliest_yr), "Scenario"].values[0]
            early_name  = earliest_sc.lstrip("🔴🟡🟢🔵 ").split("(")[0].strip()
            st.markdown(insight_card(
                f"Under <b>{early_name}</b>, global warming reaches the <b>1.5 °C threshold</b> as early as <b>{earliest_yr}</b>.",
                "amber",
            ), unsafe_allow_html=True)

with c2:
    # Scenario comparison summary: best vs worst at 2100
    temp_2100 = {}
    for sc_name in all_sc["scenario"].unique():
        row = all_sc[(all_sc["scenario"] == sc_name) & (all_sc["year"] == 2100)]
        if not row.empty:
            temp_2100[sc_name] = row["temperature_anomaly"].values[0]
    if temp_2100:
        best_sc  = min(temp_2100, key=temp_2100.get)
        worst_sc = max(temp_2100, key=temp_2100.get)
        diff     = temp_2100[worst_sc] - temp_2100[best_sc]
        best_short  = best_sc.lstrip("🔴🟡🟢🔵 ").split("(")[0].strip()
        worst_short = worst_sc.lstrip("🔴🟡🟢🔵 ").split("(")[0].strip()
        st.markdown(insight_card(
            f"The spread between <b>best-case</b> ({best_short}: {temp_2100[best_sc]:+.2f} °C) and "
            f"<b>worst-case</b> ({worst_short}: {temp_2100[worst_sc]:+.2f} °C) in 2100 is "
            f"<b>{diff:.2f} °C</b>.",
            "coral",
        ), unsafe_allow_html=True)

# ── Scenario Comparison Summary table ─────────────────────────────────────────
section_label("Scenario Comparison Summary — 2100")

milestones_yrs = [2030, 2050, 2075, 2100]
rows = []
for sc_name, sc_cfg in SCENARIOS.items():
    sc_data = all_sc[all_sc["scenario"] == sc_name].sort_values("year")
    row = {"Scenario": sc_name.lstrip("🔴🟡🟢🔵 ").split("(")[0].strip()}
    for yr in milestones_yrs:
        match = sc_data[sc_data["year"] == yr]
        row[str(yr)] = f"{match['temperature_anomaly'].values[0]:+.2f}" if not match.empty else "N/A"
    rows.append(row)

summary_tbl = pd.DataFrame(rows)
st.dataframe(summary_tbl, width="stretch", hide_index=True)
