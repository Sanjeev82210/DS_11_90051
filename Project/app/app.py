"""
Climate Intelligence Platform — Executive Overview
"""
import sys, os
_APP_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_APP_DIR, '..', 'src')))
sys.path.insert(0, _APP_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from features import create_features
from styles import DS, SERIES_COLORS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout, add_thresholds

st.set_page_config(
    page_title="Climate Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_css()

# ── Hero banner (home-only, slightly richer gradient) ─────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(105deg, #0a1628 0%, #0d2137 50%, #0a1e30 100%);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 12px;
    padding: 2rem 2.2rem 1.6rem;
    margin-bottom: 1.6rem;
">
    <div style="font-size:0.72rem;font-weight:600;color:#22d3ee;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:0.5rem;">
        Data Science Pipeline · 1958 – 2017
    </div>
    <h1 style="font-size:2.1rem;font-weight:800;color:#f1f5f9;margin:0 0 0.5rem 0;letter-spacing:-0.5px;">
        Climate Intelligence Platform
    </h1>
    <p style="font-size:1rem;color:#94a3b8;margin:0;max-width:580px;">
        Historical analysis, machine-learning forecasting, and scenario simulation
        of global temperature anomalies and CO&#8322; emissions.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/climate_data.csv")
    return create_features(df)

df = load_data()
latest  = df.iloc[-1]
earliest = df.iloc[0]
total_warming = latest["temperature_anomaly"] - earliest["temperature_anomaly"]
co2_rise = latest["co2_ppm"] - earliest["co2_ppm"]

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card(
        "Current Temp Anomaly",
        f"{latest['temperature_anomaly']:+.2f} °C",
        f"+{total_warming:.2f} °C since {int(earliest['year'])}",
        "up",
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(
        "Atmospheric CO\u2082",
        f"{latest['co2_ppm']:.1f} ppm",
        f"+{co2_rise:.1f} ppm since {int(earliest['year'])}",
        "up",
    ), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card(
        "Data Coverage",
        f"{int(earliest['year'])} – {int(latest['year'])}",
        f"{len(df)} annual observations",
        "neutral",
    ), unsafe_allow_html=True)
with k4:
    warmest_year = int(df.loc[df["temperature_anomaly"].idxmax(), "year"])
    warmest_temp = df["temperature_anomaly"].max()
    st.markdown(kpi_card(
        "Warmest Year on Record",
        str(warmest_year),
        f"{warmest_temp:+.3f} °C anomaly",
        "up",
    ), unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# ── Main climate trajectory chart ─────────────────────────────────────────────
section_label("Climate Trajectory")

fig = go.Figure()

# Temperature anomaly
fig.add_trace(go.Scatter(
    x=df["year"], y=df["temperature_anomaly"],
    name="Annual Anomaly",
    mode="lines",
    line=dict(color="rgba(244,63,94,0.3)", width=1.2),
    showlegend=True,
))
fig.add_trace(go.Scatter(
    x=df["year"], y=df["temp_10yr_avg"],
    name="10-yr Average",
    mode="lines",
    line=dict(color=DS["coral"], width=2.5),
))

# CO2 on secondary axis
fig.add_trace(go.Scatter(
    x=df["year"], y=df["co2_ppm"],
    name="CO\u2082 (ppm)",
    mode="lines",
    line=dict(color=DS["cyan"], width=2),
    yaxis="y2",
    fill="tozeroy",
    fillcolor="rgba(34,211,238,0.05)",
))

# Warmest year annotation
fig.add_annotation(
    x=warmest_year,
    y=warmest_temp,
    text=f"Warmest: {warmest_year}",
    showarrow=True,
    arrowhead=2,
    arrowcolor=DS["coral"],
    font=dict(size=10, color=DS["coral"]),
    bgcolor="rgba(15,25,35,0.85)",
    bordercolor=DS["coral"],
    borderwidth=1,
    yshift=10,
)

add_thresholds(fig)

fig.update_layout(
    **chart_layout(
        title=dict(text="Global Temperature Anomaly & Atmospheric CO\u2082 (1958–2017)"),
        height=400,
        yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis2=dict(title="CO\u2082 (ppm)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)", showgrid=False),
    )
)

st.plotly_chart(fig, width="stretch")

# ── What changed panel ────────────────────────────────────────────────────────
section_label("Key Takeaways")

rate_per_decade = (total_warming / (int(latest["year"]) - int(earliest["year"]))) * 10
corr = df["temperature_anomaly"].corr(df["co2_ppm"])

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(insight_card(
        f"<b>CO\u2082 concentration</b> has increased by <b>{co2_rise:.1f} ppm</b> since {int(earliest['year'])}, "
        f"from {earliest['co2_ppm']:.1f} to {latest['co2_ppm']:.1f} ppm.",
        "cyan",
    ), unsafe_allow_html=True)
with col_b:
    st.markdown(insight_card(
        f"<b>Temperature</b> has risen <b>{total_warming:+.2f} °C</b> over the observation period — "
        f"roughly <b>{rate_per_decade:.3f} °C per decade</b> on average.",
        "coral",
    ), unsafe_allow_html=True)
with col_c:
    st.markdown(insight_card(
        f"CO\u2082 and temperature anomaly share a <b>Pearson correlation of {corr:.3f}</b>, "
        f"one of the strongest signals in the dataset.",
        "amber",
    ), unsafe_allow_html=True)

# ── Navigation tiles ──────────────────────────────────────────────────────────
section_label("Explore")

tiles = [
    ("Historical Trends",      "Temperature & CO\u2082 time series, decade analysis, milestone timeline"),
    ("CO\u2082 Analysis",      "Correlation, scatter analysis, country comparison (218 nations)"),
    ("Model Comparison",       "Cross-validated ML leaderboard, feature importance, SHAP explainability"),
    ("Future Forecast",        "Scenario overlay, policy target tracker, 2100 projections"),
    ("Scenario Simulator",     "Custom emission pathways, uncertainty bands, saved presets"),
    ("Data Explorer",          "Quality badges, distributions, anomaly detection, CSV download"),
]

r1, r2 = st.columns(3), st.columns(3)
for i, (title, desc) in enumerate(tiles):
    col = r1[i] if i < 3 else r2[i - 3]
    with col:
        st.markdown(
            f'<div class="explore-tile">'
            f'<div class="explore-tile-title">{title}</div>'
            f'<div class="explore-tile-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);
            font-size:0.72rem;color:#475569;text-align:center;">
    Data sources: HadCRUT5 Temperature Anomaly · Our World in Data CO&#8322; Emissions · NOAA Mauna Loa CO&#8322;
    &nbsp;|&nbsp; Built with Streamlit &amp; XGBoost
</div>
""", unsafe_allow_html=True)
