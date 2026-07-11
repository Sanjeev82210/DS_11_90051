"""
Climate Intelligence Platform — Scenario Simulator
Saved presets · Uncertainty bands · Executive/Technical toggle · CSV export
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
from styles import DS, SERIES_COLORS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout, add_thresholds, warning_box

st.set_page_config(page_title="Scenario Simulator | Climate Platform", page_icon="🎛️", layout="wide")
apply_global_css()

# ── Session state init ────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}

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

df = load_data()
model = load_model(df)
features = get_feature_list()
last_hist_year = int(df["year"].max())
current_temp   = df["temperature_anomaly"].iloc[-1]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;">View Mode</div>', unsafe_allow_html=True)
    view_mode = st.radio("View", ["Technical", "Executive"], label_visibility="collapsed", horizontal=True)

    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;">Scenario</div>', unsafe_allow_html=True)

    preset_options = list(SCENARIOS.keys()) + ["Custom"]
    selected_preset = st.selectbox("Preset", preset_options, label_visibility="collapsed")

    # Auto-set slider from preset
    if selected_preset != "Custom":
        default_change = SCENARIOS[selected_preset]["annual_co2_change"] * 100
    else:
        default_change = 0.0

    annual_change_pct = st.slider(
        "Annual emissions change (%)",
        min_value=-10.0, max_value=5.0,
        value=float(round(default_change, 1)),
        step=0.1,
        help="Positive = growing emissions; negative = reduction",
    )

    # Detect custom override
    if selected_preset != "Custom":
        preset_val = round(SCENARIOS[selected_preset]["annual_co2_change"] * 100, 1)
        if round(annual_change_pct, 1) != preset_val:
            selected_preset = "Custom"

    forecast_year = st.slider("Forecast until", 2025, 2100, 2075)

    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;">Uncertainty Band</div>', unsafe_allow_html=True)
    uncertainty = st.radio("Band", ["None", "Low (±0.1°C)", "Medium (±0.2°C)", "High (±0.3°C)"],
                           label_visibility="collapsed")
    uncertainty_map = {"None": 0, "Low (±0.1°C)": 0.1, "Medium (±0.2°C)": 0.2, "High (±0.3°C)": 0.3}
    band_width = uncertainty_map[uncertainty]

    # Saved presets management
    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;">Saved Presets</div>', unsafe_allow_html=True)
    preset_name_input = st.text_input("Name this scenario", placeholder="e.g. My optimistic path", label_visibility="collapsed")
    if st.button("Save", use_container_width=True):
        if preset_name_input.strip():
            st.session_state.saved_scenarios[preset_name_input.strip()] = {
                "annual_change_pct": annual_change_pct,
                "forecast_year": forecast_year,
                "band": uncertainty,
            }
            st.success(f"Saved: {preset_name_input.strip()}")

    if st.session_state.saved_scenarios:
        saved_names = list(st.session_state.saved_scenarios.keys())
        load_name = st.selectbox("Load saved", [""] + saved_names, label_visibility="collapsed")
        if load_name:
            saved = st.session_state.saved_scenarios[load_name]
            st.caption(f"Change: {saved['annual_change_pct']:+.1f}%/yr · Until: {saved['forecast_year']} · {saved['band']}")

# ── Compute projection ────────────────────────────────────────────────────────
annual_change = annual_change_pct / 100.0
future_df = generate_scenario(df, last_hist_year + 1, forecast_year, annual_change)
preds = model.predict(future_df[features])
future_df["temperature_anomaly"] = preds

proj_temp_end = preds[-1]
proj_co2_end  = future_df["co2_ppm"].iloc[-1]
temp_delta    = proj_temp_end - current_temp

# ── Header & KPIs ─────────────────────────────────────────────────────────────
label = selected_preset if selected_preset != "Custom" else "Custom Pathway"
label_clean = label.lstrip("🔴🟡🟢🔵 ").split("(")[0].strip() if selected_preset != "Custom" else "Custom"

page_header("Scenario Simulator", "Adjust emission pathway, set uncertainty bands, and compare against all preset scenarios")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Scenario", label_clean, f"{annual_change_pct:+.1f}%/yr emissions change", "neutral"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(f"Temp in {forecast_year}", f"{proj_temp_end:+.2f} °C", f"vs {current_temp:+.2f} °C now", "up"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card(f"CO\u2082 (ppm) in {forecast_year}", f"{proj_co2_end:.1f} ppm", "Atmospheric concentration", "up"), unsafe_allow_html=True)
with k4:
    delta_dir = "up" if temp_delta > 0 else "down"
    st.markdown(kpi_card("Change from Now", f"{temp_delta:+.2f} °C", f"by {forecast_year}", delta_dir), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

# ── Main forecast chart ───────────────────────────────────────────────────────
section_label("Temperature Projection")

# Get all preset scenarios as background context
all_sc_bg = compare_scenarios(df, forecast_year, features, model)

fig = go.Figure()

# Background preset lines (semi-transparent)
for sc_name, sc_cfg in SCENARIOS.items():
    sc_data = all_sc_bg[all_sc_bg["scenario"] == sc_name].sort_values("year")
    if sc_data.empty:
        continue
    short = sc_name.lstrip("🔴🟡🟢🔵 ").split("(")[0].strip()
    fig.add_trace(go.Scatter(
        x=sc_data["year"], y=sc_data["temperature_anomaly"],
        mode="lines", name=short,
        line=dict(color=sc_cfg["color"], width=1.2, dash="dot"),
        opacity=0.35,
        hovertemplate=f"<b>{short}</b><br>Year: %{{x}}<br>Temp: %{{y:.3f}} °C<extra></extra>",
    ))

# Historical
fig.add_trace(go.Scatter(
    x=df["year"], y=df["temperature_anomaly"],
    name="Historical", mode="lines",
    line=dict(color=DS["cyan"], width=2.2),
))

# Uncertainty band
if band_width > 0:
    fig.add_trace(go.Scatter(
        x=list(future_df["year"]) + list(future_df["year"])[::-1],
        y=list(future_df["temperature_anomaly"] + band_width) + list(future_df["temperature_anomaly"] - band_width)[::-1],
        fill="toself", fillcolor="rgba(34,211,238,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"\u00b1{band_width} °C band", showlegend=True,
        hoverinfo="skip",
    ))

# Selected scenario (bold)
fig.add_trace(go.Scatter(
    x=future_df["year"], y=future_df["temperature_anomaly"],
    name=f"Selected: {label_clean}", mode="lines",
    line=dict(color=DS["amber"], width=3),
    hovertemplate="Year: %{x}<br>Projected: %{y:.3f} °C<extra></extra>",
))

fig.add_vline(
    x=last_hist_year, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"),
    annotation=dict(text="Forecast start", font=dict(size=10, color="#94a3b8"), xanchor="left"),
)
add_thresholds(fig)

fig.update_layout(**chart_layout(
    title=dict(text=f"Temperature Projection: {label_clean} (until {forecast_year})"),
    height=440,
    yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.08)"),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig, width="stretch")

# ── Executive insight / Technical detail ──────────────────────────────────────
if view_mode == "Executive":
    section_label("Key Takeaway")
    st.markdown(insight_card(
        f"Under the <b>{label_clean}</b> pathway, global temperature anomaly reaches <b>{proj_temp_end:+.2f} °C</b> by {forecast_year}, "
        f"a change of <b>{temp_delta:+.2f} °C</b> from today's {current_temp:+.2f} °C.",
        "cyan" if temp_delta < 0.5 else ("amber" if temp_delta < 1.0 else "coral"),
    ), unsafe_allow_html=True)
else:
    # Technical: CO2 pathway charts
    section_label("CO\u2082 Pathways")
    col_l, col_r = st.columns(2)
    with col_l:
        fig_em = go.Figure()
        fig_em.add_trace(go.Scatter(
            x=df["year"], y=df["co2_emissions"], name="Historical",
            mode="lines", line=dict(color=DS["cyan"], width=2),
        ))
        fig_em.add_trace(go.Scatter(
            x=future_df["year"], y=future_df["co2_emissions"], name="Projected",
            mode="lines", line=dict(color=DS["coral"], width=2, dash="dash"),
        ))
        fig_em.add_vline(x=last_hist_year, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))
        fig_em.update_layout(**chart_layout(
            title=dict(text="CO\u2082 Emissions"), height=320,
            yaxis=dict(title="Million Tonnes", gridcolor="rgba(255,255,255,0.05)"),
        ))
        st.plotly_chart(fig_em, width="stretch")

    with col_r:
        fig_ppm = go.Figure()
        fig_ppm.add_trace(go.Scatter(
            x=df["year"], y=df["co2_ppm"], name="Historical",
            mode="lines", line=dict(color=DS["cyan"], width=2),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.05)",
        ))
        fig_ppm.add_trace(go.Scatter(
            x=future_df["year"], y=future_df["co2_ppm"], name="Projected",
            mode="lines", line=dict(color=DS["amber"], width=2, dash="dash"),
        ))
        fig_ppm.add_vline(x=last_hist_year, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))
        fig_ppm.update_layout(**chart_layout(
            title=dict(text="Atmospheric CO\u2082 (ppm)"), height=320,
            yaxis=dict(title="ppm", gridcolor="rgba(255,255,255,0.05)"),
        ))
        st.plotly_chart(fig_ppm, width="stretch")

    # Year-by-year table in expander
    with st.expander("Year-by-year projection data"):
        table_df = future_df[["year", "co2_emissions", "co2_ppm", "temperature_anomaly"]].copy()
        table_df.columns = ["Year", "CO\u2082 Emissions (Mt)", "CO\u2082 (ppm)", "Projected Temp (°C)"]
        st.dataframe(
            table_df.style.format({
                "CO\u2082 Emissions (Mt)": "{:,.1f}",
                "CO\u2082 (ppm)": "{:.2f}",
                "Projected Temp (°C)": "{:+.4f}",
            }),
            width="stretch", hide_index=True,
        )

# ── CSV download (always available) ──────────────────────────────────────────
csv_bytes = future_df[["year", "co2_emissions", "co2_ppm", "temperature_anomaly"]].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download projection as CSV",
    data=csv_bytes,
    file_name=f"climate_projection_{label_clean.replace(' ', '_').lower()}_{forecast_year}.csv",
    mime="text/csv",
)

# ── Saved scenarios summary ───────────────────────────────────────────────────
if st.session_state.saved_scenarios and view_mode == "Technical":
    section_label("Your Saved Scenarios")
    saved_rows = [
        {
            "Name": n,
            "Emissions Change": f"{v['annual_change_pct']:+.1f}%/yr",
            "Until": v["forecast_year"],
            "Uncertainty": v["band"],
        }
        for n, v in st.session_state.saved_scenarios.items()
    ]
    st.dataframe(pd.DataFrame(saved_rows), width="stretch", hide_index=True)
