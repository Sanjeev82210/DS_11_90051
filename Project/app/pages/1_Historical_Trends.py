"""
Climate Intelligence Platform — Historical Trends
"""
import sys, os
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
_APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _SRC)
sys.path.insert(0, _APP)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from features import create_features
from styles import DS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout, add_thresholds, warning_box

st.set_page_config(page_title="Historical Trends | Climate Platform", page_icon="📈", layout="wide")
apply_global_css()

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/climate_data.csv")
    return create_features(df)

df = load_data()
latest   = df.iloc[-1]
earliest = df.iloc[0]
warmest_idx  = df["temperature_anomaly"].idxmax()
warmest_year = int(df.loc[warmest_idx, "year"])
warmest_temp = df.loc[warmest_idx, "temperature_anomaly"]
total_warming = latest["temperature_anomaly"] - earliest["temperature_anomaly"]

# ── Milestones ────────────────────────────────────────────────────────────────
milestones = []
for ppm_target in [350, 400]:
    crossed = df[df["co2_ppm"] >= ppm_target]
    if not crossed.empty:
        yr = int(crossed.iloc[0]["year"])
        milestones.append({"year": yr, "label": f"CO\u2082 \u2265 {ppm_target} ppm", "color": DS["amber"] if ppm_target == 350 else DS["coral"]})
milestones.append({"year": warmest_year, "label": f"Warmest year: {warmest_year}", "color": DS["coral"]})
milestones.append({"year": int(latest["year"]) + 1, "label": "Forecast start", "color": DS["cyan"]})

# ── Header & KPIs ─────────────────────────────────────────────────────────────
page_header("Historical Trends", "Temperature anomalies, CO\u2082 concentration, rolling averages, and decade-level analysis")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Latest Temp Anomaly", f"{latest['temperature_anomaly']:+.2f} °C", f"Year {int(latest['year'])}", "up"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Warmest Year", str(warmest_year), f"{warmest_temp:+.2f} °C anomaly", "up"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("CO\u2082 (Latest)", f"{latest['co2_ppm']:.1f} ppm", f"+{latest['co2_ppm'] - earliest['co2_ppm']:.1f} ppm rise", "up"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Total Warming", f"{total_warming:+.2f} °C", f"{int(earliest['year'])} \u2192 {int(latest['year'])}", "up"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

# ── Side-by-side charts ───────────────────────────────────────────────────────
section_label("Temperature & CO\u2082 Concentration")
col_l, col_r = st.columns(2)

with col_l:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df["year"], y=df["temperature_anomaly"], name="Annual",
        mode="lines", line=dict(color="rgba(244,63,94,0.25)", width=1.2),
    ))
    fig_temp.add_trace(go.Scatter(
        x=df["year"], y=df["temp_5yr_avg"], name="5-yr avg",
        mode="lines", line=dict(color=DS["amber"], width=2.2),
    ))
    fig_temp.add_trace(go.Scatter(
        x=df["year"], y=df["temp_10yr_avg"], name="10-yr avg",
        mode="lines", line=dict(color=DS["coral"], width=2.6),
    ))
    fig_temp.add_annotation(
        x=warmest_year, y=warmest_temp,
        text=f"{warmest_year}", showarrow=True, arrowhead=2,
        arrowcolor=DS["coral"], font=dict(size=10, color=DS["coral"]),
        bgcolor="rgba(15,25,35,0.85)", bordercolor=DS["coral"], borderwidth=1, yshift=8,
    )
    add_thresholds(fig_temp)
    fig_temp.update_layout(**chart_layout(
        title=dict(text="Temperature Anomaly (°C)"), height=380,
        yaxis=dict(title="°C", gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
    ))
    st.plotly_chart(fig_temp, width="stretch")

with col_r:
    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(
        x=df["year"], y=df["co2_ppm"], name="CO\u2082 (ppm)",
        mode="lines", line=dict(color=DS["cyan"], width=2.2),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.07)",
    ))
    # Annotate 350 and 400 ppm crossings
    for m in milestones[:2]:
        yr_row = df[df["year"] == m["year"]]
        if not yr_row.empty:
            fig_co2.add_annotation(
                x=m["year"], y=yr_row["co2_ppm"].values[0],
                text=m["label"].split(":")[0], showarrow=True, arrowhead=2,
                arrowcolor=m["color"], font=dict(size=9, color=m["color"]),
                bgcolor="rgba(15,25,35,0.85)", bordercolor=m["color"], borderwidth=1,
            )
    fig_co2.update_layout(**chart_layout(
        title=dict(text="Atmospheric CO\u2082 Concentration"), height=380,
        yaxis=dict(title="ppm", gridcolor="rgba(255,255,255,0.05)"),
    ))
    st.plotly_chart(fig_co2, width="stretch")

# ── Combined dual-axis ────────────────────────────────────────────────────────
section_label("Combined View")
fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
fig_dual.add_trace(go.Scatter(
    x=df["year"], y=df["temperature_anomaly"], name="Temp Anomaly",
    mode="lines+markers", line=dict(color=DS["coral"], width=1.8),
    marker=dict(size=3),
), secondary_y=False)
fig_dual.add_trace(go.Scatter(
    x=df["year"], y=df["temp_10yr_avg"], name="10-yr Avg",
    mode="lines", line=dict(color=DS["amber"], width=2.2),
), secondary_y=False)
fig_dual.add_trace(go.Scatter(
    x=df["year"], y=df["co2_ppm"], name="CO\u2082 (ppm)",
    mode="lines", line=dict(color=DS["cyan"], width=2),
    fill="tozeroy", fillcolor="rgba(34,211,238,0.05)",
), secondary_y=True)
fig_dual.update_layout(**chart_layout(
    title=dict(text="Temperature Anomaly vs Atmospheric CO\u2082"), height=400,
))
fig_dual.update_yaxes(title_text="Temperature Anomaly (°C)", secondary_y=False, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
fig_dual.update_yaxes(title_text="CO\u2082 (ppm)", secondary_y=True, gridcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False)
st.plotly_chart(fig_dual, width="stretch")

# ── Insight cards ─────────────────────────────────────────────────────────────
section_label("Key Insights")
co2_rise = latest["co2_ppm"] - earliest["co2_ppm"]
rate = (total_warming / (int(latest["year"]) - int(earliest["year"]))) * 10
corr = df["temperature_anomaly"].corr(df["co2_ppm"])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(insight_card(f"The 10-year rolling average reveals a clear acceleration in warming post-1980, with every decade warmer than the last.", "cyan"), unsafe_allow_html=True)
with c2:
    st.markdown(insight_card(f"CO\u2082 concentration rose <b>{co2_rise:.1f} ppm</b> over the observation window — an increase of <b>{co2_rise/earliest['co2_ppm']*100:.1f}%</b>.", "coral"), unsafe_allow_html=True)
with c3:
    st.markdown(insight_card(f"The Pearson correlation between CO\u2082 ppm and temperature anomaly is <b>{corr:.3f}</b>, indicating a very strong positive relationship.", "amber"), unsafe_allow_html=True)

# ── Milestone timeline ────────────────────────────────────────────────────────
section_label("Climate Milestone Timeline")

mile_years  = [m["year"] for m in milestones]
mile_labels = [m["label"] for m in milestones]
mile_colors = [m["color"] for m in milestones]

fig_time = go.Figure()
fig_time.add_shape(type="line", x0=min(mile_years)-2, x1=max(mile_years)+2, y0=0, y1=0,
                   line=dict(color="rgba(255,255,255,0.12)", width=2))
for i, (yr, lbl, col) in enumerate(zip(mile_years, mile_labels, mile_colors)):
    yoff = 0.3 if i % 2 == 0 else -0.3
    fig_time.add_trace(go.Scatter(
        x=[yr], y=[0],
        mode="markers+text",
        marker=dict(size=12, color=col, line=dict(width=2, color="rgba(255,255,255,0.3)")),
        text=[str(yr)],
        textposition="top center",
        textfont=dict(size=10, color=col),
        showlegend=False,
        hovertemplate=f"<b>{lbl}</b><br>Year: {yr}<extra></extra>",
    ))
    fig_time.add_annotation(
        x=yr, y=yoff, text=lbl,
        showarrow=True, arrowhead=0, ax=0, ay=-20 if yoff > 0 else 20,
        font=dict(size=9.5, color=col), bgcolor="rgba(15,25,35,0.85)",
        bordercolor=col, borderwidth=1, borderpad=3,
    )

fig_time.update_layout(**chart_layout(
    title=dict(text="Key Climate Milestones"),
    height=220,
    xaxis=dict(showgrid=False, zeroline=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 0.8], title=""),
    margin=dict(l=20, r=20, t=44, b=20),
))
st.plotly_chart(fig_time, width="stretch")

# ── Decade analysis ───────────────────────────────────────────────────────────
section_label("Decade Analysis")
decade_df = df.groupby("decade").agg(
    avg_temp=("temperature_anomaly", "mean"),
    avg_co2=("co2_ppm", "mean"),
    count=("year", "count"),
).reset_index()
decade_df["decade_label"] = decade_df["decade"].astype(str) + "s"

max_temp = decade_df["avg_temp"].max()
bar_colors = [DS["coral"] if v == max_temp else DS["cyan"] for v in decade_df["avg_temp"]]

fig_dec = go.Figure()
fig_dec.add_trace(go.Bar(
    x=decade_df["decade_label"], y=decade_df["avg_temp"],
    marker=dict(color=bar_colors, cornerradius=4),
    text=[f"{v:+.3f} °C" for v in decade_df["avg_temp"]],
    textposition="outside", textfont=dict(color="#94a3b8", size=11),
    hovertemplate="<b>%{x}</b><br>Avg anomaly: %{y:.4f} °C<extra></extra>",
))
fig_dec.update_layout(**chart_layout(
    title=dict(text="Mean Temperature Anomaly by Decade"), height=360, showlegend=False,
    yaxis=dict(title="°C", gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
))
st.plotly_chart(fig_dec, width="stretch")

with st.expander("Decade statistics table"):
    st.dataframe(
        decade_df.rename(columns={
            "decade_label": "Decade", "avg_temp": "Avg Temp Anomaly (°C)",
            "avg_co2": "Avg CO\u2082 (ppm)", "count": "Years",
        }).drop(columns=["decade"]).style.format({
            "Avg Temp Anomaly (°C)": "{:+.4f}", "Avg CO\u2082 (ppm)": "{:.1f}",
        }),
        width="stretch", hide_index=True,
    )
