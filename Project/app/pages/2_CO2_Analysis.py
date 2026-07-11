"""
Climate Intelligence Platform — CO2 Analysis & Country Comparison
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
import plotly.express as px
from features import create_features
from preprocessing import load_co2_country_data
from styles import DS, SERIES_COLORS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout, warning_box

st.set_page_config(page_title="CO\u2082 Analysis | Climate Platform", page_icon="📊", layout="wide")
apply_global_css()

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/climate_data.csv")
    return create_features(df)

@st.cache_data
def load_countries():
    return load_co2_country_data("data/raw/co2_emissions.csv")

df = load_data()
country_df = load_countries()

latest   = df.iloc[-1]
earliest = df.iloc[0]
corr_val = df["temperature_anomaly"].corr(df["co2_ppm"])
co2_growth_latest = df["co2_growth"].iloc[-1]
cumulative_total = df["cumulative_co2"].iloc[-1]

# Simple OLS R2
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(df["co2_ppm"], df["temperature_anomaly"], 1)
y_hat  = np.polyval(coeffs, df["co2_ppm"])
ss_res = np.sum((df["temperature_anomaly"] - y_hat) ** 2)
ss_tot = np.sum((df["temperature_anomaly"] - df["temperature_anomaly"].mean()) ** 2)
r2_val = 1 - ss_res / ss_tot

# ── Header & KPIs ─────────────────────────────────────────────────────────────
page_header("CO\u2082 Analysis", "Correlation analysis, scatter plots, and country-level emission comparisons")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Pearson Correlation", f"{corr_val:.3f}", "CO\u2082 ppm vs Temperature", "up"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("OLS R\u00b2", f"{r2_val:.3f}", "Explained variance", "neutral"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("CO\u2082 Growth Rate", f"{co2_growth_latest:+.2f}%", "Latest year-over-year", "up"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Cumulative Emissions", f"{cumulative_total/1000:.1f} Gt", "Total since baseline", "up"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

# ── Scatter: CO2 ppm vs Temperature ──────────────────────────────────────────
section_label("CO\u2082 vs Temperature Correlation")
col_l, col_r = st.columns([3, 2])

with col_l:
    # Add OLS trendline manually
    x_range = np.linspace(df["co2_ppm"].min(), df["co2_ppm"].max(), 100)
    y_trend = np.polyval(coeffs, x_range)

    norm_years = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=df["co2_ppm"], y=df["temperature_anomaly"],
        mode="markers",
        marker=dict(
            size=9,
            color=norm_years,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text="Year", font=dict(size=11)), thickness=12, len=0.7),
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        text=df["year"].astype(int),
        hovertemplate="Year: <b>%{text}</b><br>CO\u2082: %{x:.1f} ppm<br>Temp: %{y:.3f} °C<extra></extra>",
        name="Observations",
    ))
    fig_sc.add_trace(go.Scatter(
        x=x_range, y=y_trend,
        mode="lines", name=f"OLS Trend (R\u00b2={r2_val:.3f})",
        line=dict(color=DS["coral"], width=2, dash="dash"),
    ))
    fig_sc.update_layout(**chart_layout(
        title=dict(text="Atmospheric CO\u2082 vs Temperature Anomaly"),
        height=420,
        xaxis=dict(title="CO\u2082 (ppm)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)"),
    ))
    st.plotly_chart(fig_sc, width="stretch")

with col_r:
    # Correlation heatmap
    corr_cols = ["temperature_anomaly", "co2_ppm", "co2_emissions", "cumulative_co2"]
    corr_labels = ["Temp Anomaly", "CO\u2082 (ppm)", "CO\u2082 Emissions", "Cumulative CO\u2082"]
    corr_matrix = df[corr_cols].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_labels, y=corr_labels,
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont=dict(size=13, color="#f1f5f9"),
        hovertemplate="%{x} vs %{y}: <b>%{z:.3f}</b><extra></extra>",
    ))
    fig_heat.update_layout(**chart_layout(
        title=dict(text="Correlation Matrix"), height=420,
        margin=dict(l=100, r=20, t=44, b=80),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
    ))
    st.plotly_chart(fig_heat, width="stretch")

# ── Insight cards ─────────────────────────────────────────────────────────────
section_label("Correlation Insights")
co2_rise = latest["co2_ppm"] - earliest["co2_ppm"]
c1, c2 = st.columns(2)
with c1:
    st.markdown(insight_card(
        f"CO\u2082 concentration and temperature share a Pearson correlation of <b>{corr_val:.3f}</b>. "
        f"The OLS regression explains <b>{r2_val*100:.1f}%</b> of temperature variance with CO\u2082 as the sole predictor.",
        "cyan",
    ), unsafe_allow_html=True)
with c2:
    st.markdown(insight_card(
        f"Cumulative emissions — the running total of CO\u2082 released — show an even stronger relationship with temperature, "
        f"supporting the physical understanding that warming is driven by the stock, not just the flow, of atmospheric CO\u2082.",
        "amber",
    ), unsafe_allow_html=True)

warning_box(
    "<b>Correlation \u2260 Causation</b> &mdash; This analysis does not prove the causal mechanism. "
    "The physical link between CO\u2082 forcing and warming is established by climate science independently of these correlations."
)

# ── Cumulative emissions scatter ──────────────────────────────────────────────
section_label("Cumulative Emissions vs Temperature")
coeffs2 = np.polyfit(df["cumulative_co2"], df["temperature_anomaly"], 1)
x2_range = np.linspace(df["cumulative_co2"].min(), df["cumulative_co2"].max(), 100)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=df["cumulative_co2"], y=df["temperature_anomaly"],
    mode="markers",
    marker=dict(size=8, color=DS["violet"], opacity=0.8, line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
    text=df["year"].astype(int),
    hovertemplate="Year: <b>%{text}</b><br>Cumulative: %{x:,.0f} Mt<br>Temp: %{y:.3f} °C<extra></extra>",
    name="Observations",
))
fig_cum.add_trace(go.Scatter(
    x=x2_range, y=np.polyval(coeffs2, x2_range),
    mode="lines", name="OLS Trend",
    line=dict(color=DS["amber"], width=2, dash="dash"),
))
fig_cum.update_layout(**chart_layout(
    title=dict(text="Cumulative CO\u2082 Emissions vs Temperature Anomaly"),
    height=360,
    xaxis=dict(title="Cumulative CO\u2082 Emissions (Million Tonnes)", gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig_cum, width="stretch")

# ── Country Comparison Mode ───────────────────────────────────────────────────
section_label("Country Comparison")
st.markdown(
    '<p style="font-size:0.82rem;color:#94a3b8;margin-bottom:0.8rem;">'
    'Select 2–5 countries to compare annual emissions, per-capita emissions, and global share.</p>',
    unsafe_allow_html=True,
)

available_countries = sorted(country_df["country"].dropna().unique().tolist())
default_countries   = ["China", "United States", "India", "Russia", "Germany"]
default_countries   = [c for c in default_countries if c in available_countries]

selected_countries = st.multiselect(
    "Countries",
    options=available_countries,
    default=default_countries[:4],
    max_selections=5,
)

if selected_countries:
    year_range = st.slider(
        "Year range",
        int(country_df["year"].min()), int(country_df["year"].max()),
        (1970, int(country_df["year"].max())),
    )

    cdf = country_df[
        (country_df["country"].isin(selected_countries)) &
        (country_df["year"].between(*year_range)) &
        (country_df["co2_emissions"].notna())
    ]

    # Compute global totals per year for share calculation
    global_annual = country_df.groupby("year")["co2_emissions"].sum().reset_index()
    global_annual = global_annual.rename(columns={"co2_emissions": "global_total"})
    cdf = cdf.merge(global_annual, on="year", how="left")
    cdf["share_pct"] = cdf["co2_emissions"] / cdf["global_total"] * 100

    tab1, tab2, tab3 = st.tabs(["Annual Emissions", "Per Capita", "Global Share %"])

    with tab1:
        fig_ann = go.Figure()
        for i, country in enumerate(selected_countries):
            sub = cdf[cdf["country"] == country]
            if sub.empty:
                continue
            fig_ann.add_trace(go.Scatter(
                x=sub["year"], y=sub["co2_emissions"],
                name=country, mode="lines",
                line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
                hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Emissions: %{{y:,.1f}} Mt<extra></extra>",
            ))
        fig_ann.update_layout(**chart_layout(
            title=dict(text="Annual CO\u2082 Emissions by Country"), height=380,
            yaxis=dict(title="CO\u2082 Emissions (Million Tonnes)", gridcolor="rgba(255,255,255,0.05)"),
        ))
        st.plotly_chart(fig_ann, width="stretch")

    with tab2:
        fig_pc = go.Figure()
        for i, country in enumerate(selected_countries):
            sub = cdf[cdf["country"] == country].dropna(subset=["co2_per_capita"])
            if sub.empty:
                continue
            fig_pc.add_trace(go.Scatter(
                x=sub["year"], y=sub["co2_per_capita"],
                name=country, mode="lines",
                line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
                hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Per capita: %{{y:.2f}} t/person<extra></extra>",
            ))
        fig_pc.update_layout(**chart_layout(
            title=dict(text="CO\u2082 Emissions Per Capita"), height=380,
            yaxis=dict(title="Tonnes per Person", gridcolor="rgba(255,255,255,0.05)"),
        ))
        st.plotly_chart(fig_pc, width="stretch")

    with tab3:
        fig_share = go.Figure()
        for i, country in enumerate(selected_countries):
            sub = cdf[cdf["country"] == country].dropna(subset=["share_pct"])
            if sub.empty:
                continue
            fig_share.add_trace(go.Scatter(
                x=sub["year"], y=sub["share_pct"],
                name=country, mode="lines",
                line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
                hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Share: %{{y:.2f}}%<extra></extra>",
            ))
        fig_share.update_layout(**chart_layout(
            title=dict(text="Share of Global CO\u2082 Emissions (%)"), height=380,
            yaxis=dict(title="% of Global Emissions", gridcolor="rgba(255,255,255,0.05)"),
        ))
        st.plotly_chart(fig_share, width="stretch")

    with st.expander("Raw country data"):
        show_cols = ["country", "year", "co2_emissions", "co2_per_capita", "share_pct"]
        show_cols = [c for c in show_cols if c in cdf.columns]
        st.dataframe(
            cdf[show_cols].rename(columns={
                "co2_emissions": "Emissions (Mt)",
                "co2_per_capita": "Per Capita (t)",
                "share_pct": "Global Share (%)",
            }).style.format({
                "Emissions (Mt)": "{:,.1f}",
                "Per Capita (t)": "{:.2f}",
                "Global Share (%)": "{:.2f}",
            }),
            width="stretch", hide_index=True,
        )
else:
    st.info("Select at least one country above.")
