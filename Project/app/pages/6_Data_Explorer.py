"""
Climate Intelligence Platform — Data Explorer
Data Quality Badges · IQR anomaly detection · Distributions · CSV download
"""
import sys, os
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
_APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _SRC)
sys.path.insert(0, _APP)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from features import create_features
from preprocessing import get_data_quality_report
from styles import DS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout, badge, warning_box

st.set_page_config(page_title="Data Explorer | Climate Platform", page_icon="🔍", layout="wide")
apply_global_css()

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_csv("data/processed/climate_data.csv")
    feat = create_features(raw)
    return raw, feat

raw_df, df = load_data()
report = get_data_quality_report(raw_df)

# IQR outlier detection
Q1 = df["temperature_anomaly"].quantile(0.25)
Q3 = df["temperature_anomaly"].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df["temperature_anomaly"] < Q1 - 1.5 * IQR) | (df["temperature_anomaly"] > Q3 + 1.5 * IQR)
outlier_count = outlier_mask.sum()
last_year = int(df["year"].max())
missing_count = int(raw_df.isnull().sum().sum())

# ── Header & KPIs ─────────────────────────────────────────────────────────────
page_header("Data Explorer", "Quality inspection, distributions, anomaly detection, and raw data download")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Total Rows", str(report["total_rows"]), "Annual observations", "neutral"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Year Range", report["year_range"], "Observation window", "neutral"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Missing Values", str(missing_count), "Across all columns", "up" if missing_count > 0 else "neutral"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Temperature Outliers", str(outlier_count), "IQR-based detection", "neutral"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

# ── Data Quality Badges ───────────────────────────────────────────────────────
section_label("Data Quality")

# Determine badge colors
date_color   = "green" if last_year >= 2015 else ("amber" if last_year >= 2000 else "red")
missing_color= "green" if missing_count == 0 else ("amber" if missing_count < 50 else "red")
outlier_color= "green" if outlier_count == 0 else ("amber" if outlier_count <= 5 else "red")
source_color = "green"

badges_html = (
    badge(f"Last year: {last_year}", date_color) +
    badge(f"{missing_count} missing values", missing_color) +
    badge(f"{outlier_count} outliers detected", outlier_color) +
    badge(f"{report['total_rows']} observations", "neutral") +
    badge("3 source datasets", source_color) +
    badge("UTF-8 encoded", "green")
)
st.markdown(f'<div style="margin-bottom:1rem;">{badges_html}</div>', unsafe_allow_html=True)

# Per-column missing breakdown
with st.expander("Missing values per column"):
    miss_df = pd.DataFrame({
        "Column": raw_df.columns,
        "Missing": raw_df.isnull().sum().values,
        "% Missing": (raw_df.isnull().sum().values / len(raw_df) * 100).round(1),
    })
    st.dataframe(miss_df.style.format({"% Missing": "{:.1f}%"}), width="stretch", hide_index=True)

# ── Anomaly detection ─────────────────────────────────────────────────────────
section_label("Anomaly Detection — Temperature Anomaly")

normal_df  = df[~outlier_mask]
outlier_df = df[outlier_mask]

fig_anom = go.Figure()
fig_anom.add_hline(y=Q3 + 1.5 * IQR, line=dict(color=DS["coral"], width=1, dash="dash"),
                   annotation=dict(text="Upper IQR bound", font=dict(size=10, color=DS["coral"]), xanchor="right", x=1))
fig_anom.add_hline(y=Q1 - 1.5 * IQR, line=dict(color=DS["amber"], width=1, dash="dash"),
                   annotation=dict(text="Lower IQR bound", font=dict(size=10, color=DS["amber"]), xanchor="right", x=1))
fig_anom.add_trace(go.Scatter(
    x=normal_df["year"], y=normal_df["temperature_anomaly"],
    mode="markers", name="Normal",
    marker=dict(size=7, color=DS["cyan"], opacity=0.85, line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
    hovertemplate="Year: %{x}<br>Temp: %{y:.4f} °C<extra></extra>",
))
if not outlier_df.empty:
    fig_anom.add_trace(go.Scatter(
        x=outlier_df["year"], y=outlier_df["temperature_anomaly"],
        mode="markers", name="Outlier",
        marker=dict(size=11, color=DS["coral"], symbol="x", line=dict(width=2, color=DS["coral"])),
        hovertemplate="<b>Outlier</b><br>Year: %{x}<br>Temp: %{y:.4f} °C<extra></extra>",
    ))

fig_anom.update_layout(**chart_layout(
    title=dict(text="Temperature Anomaly — IQR Outlier Detection"), height=360,
    yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig_anom, width="stretch")

if not outlier_df.empty:
    st.markdown(insight_card(
        f"<b>{outlier_count} outlier year(s)</b> detected using IQR method (Q1-1.5×IQR to Q3+1.5×IQR): "
        f"{', '.join(outlier_df['year'].astype(int).astype(str).tolist())}. "
        f"These are kept in the model to preserve the true climate signal.",
        "coral",
    ), unsafe_allow_html=True)
else:
    st.markdown(insight_card("No outliers detected in temperature anomaly using IQR method.", "green"), unsafe_allow_html=True)

# ── Distributions ─────────────────────────────────────────────────────────────
section_label("Distributions")

col_l, col_r = st.columns(2)
with col_l:
    fig_hist_t = go.Figure()
    fig_hist_t.add_trace(go.Histogram(
        x=df["temperature_anomaly"], nbinsx=18,
        marker=dict(color=DS["coral"], opacity=0.8, line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        name="Temp Anomaly",
    ))
    fig_hist_t.update_layout(**chart_layout(
        title=dict(text="Distribution — Temperature Anomaly (°C)"), height=320,
        showlegend=False,
        xaxis=dict(title="°C"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
    ))
    st.plotly_chart(fig_hist_t, width="stretch")

with col_r:
    fig_hist_c = go.Figure()
    fig_hist_c.add_trace(go.Histogram(
        x=df["co2_ppm"], nbinsx=18,
        marker=dict(color=DS["cyan"], opacity=0.8, line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        name="CO\u2082 ppm",
    ))
    fig_hist_c.update_layout(**chart_layout(
        title=dict(text="Distribution — Atmospheric CO\u2082 (ppm)"), height=320,
        showlegend=False,
        xaxis=dict(title="ppm"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
    ))
    st.plotly_chart(fig_hist_c, width="stretch")

# ── Year-over-year change ─────────────────────────────────────────────────────
section_label("Year-over-Year Temperature Change")

yoy = df[["year", "temperature_anomaly"]].copy()
yoy["change"] = yoy["temperature_anomaly"].diff()
yoy = yoy.dropna(subset=["change"])

fig_yoy = go.Figure()
fig_yoy.add_trace(go.Bar(
    x=yoy["year"], y=yoy["change"],
    marker=dict(
        color=[DS["coral"] if v >= 0 else DS["cyan"] for v in yoy["change"]],
        cornerradius=3,
    ),
    hovertemplate="Year: %{x}<br>Change: %{y:+.4f} °C<extra></extra>",
    name="YoY Change",
))
fig_yoy.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
fig_yoy.update_layout(**chart_layout(
    title=dict(text="Annual Temperature Change (vs Previous Year)"), height=320, showlegend=False,
    yaxis=dict(title="°C change", gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig_yoy, width="stretch")

# ── Full dataset ──────────────────────────────────────────────────────────────
with st.expander("Full processed dataset"):
    st.dataframe(
        raw_df.style.format({
            "temperature_anomaly": "{:+.4f}",
            "co2_emissions": "{:,.1f}",
            "co2_per_capita": "{:.3f}",
            "co2_ppm": "{:.2f}",
        }),
        width="stretch", hide_index=True,
    )

# ── Download ──────────────────────────────────────────────────────────────────
csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download processed dataset (CSV)",
    data=csv_bytes,
    file_name="climate_processed_data.csv",
    mime="text/csv",
)
