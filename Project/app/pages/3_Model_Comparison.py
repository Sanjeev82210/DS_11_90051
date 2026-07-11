"""
Climate Intelligence Platform — Model Comparison & Explainability
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
from features import create_features, get_feature_list
from models import train_and_evaluate, get_feature_importance, train_final_model
from explainability import compute_shap_values, shap_summary_df
from styles import DS, SERIES_COLORS, apply_global_css, page_header, kpi_card, insight_card, section_label, chart_layout

st.set_page_config(page_title="Model Comparison | Climate Platform", page_icon="🤖", layout="wide")
apply_global_css()

MODEL_COLORS = {
    "Linear Regression": DS["cyan"],
    "Random Forest":     DS["green"],
    "XGBoost":           DS["amber"],
    "ARIMA":             DS["coral"],
}

# ── Data & models (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/climate_data.csv")
    return create_features(df)

@st.cache_data
def run_evaluation(_X, _y):
    return train_and_evaluate(_X, _y)

@st.cache_resource
def get_xgb_model(_X, _y):
    import joblib
    path = "models/xgboost_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return train_final_model(_X, _y, "XGBoost")

df = load_data()
features = get_feature_list()
X = df[features]
y = df["temperature_anomaly"]

with st.spinner("Running cross-validation…"):
    results_df, last_fold_data = run_evaluation(X, y)

results_df = results_df.sort_values("MAE").reset_index(drop=True)
best = results_df.iloc[0]

# ── Header & KPIs ─────────────────────────────────────────────────────────────
page_header("Model Comparison", "Cross-validated ML leaderboard, feature importance, and SHAP explainability")

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(kpi_card("Best Model", best["Model"], f"R\u00b2 = {best['R2']:.4f}", "neutral"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Best MAE", f"{best['MAE']:.4f} °C", "Mean Absolute Error", "down"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Best RMSE", f"{best['RMSE']:.4f} °C", "Root Mean Squared Error", "down"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

# ── Leaderboard ───────────────────────────────────────────────────────────────
section_label("Leaderboard")

col_tbl, col_bar = st.columns([2, 3])

with col_tbl:
    ranked = results_df.copy()
    ranked.index = range(1, len(ranked) + 1)
    ranked.index.name = "Rank"
    st.dataframe(
        ranked.style
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"})
        .background_gradient(subset=["MAE"], cmap="RdYlGn_r", vmin=0)
        .background_gradient(subset=["R2"], cmap="RdYlGn", vmin=-1, vmax=1),
        width="stretch",
    )

with col_bar:
    model_names = results_df["Model"].tolist()
    bar_colors  = [MODEL_COLORS.get(m, DS["cyan"]) for m in model_names]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="MAE", x=results_df["MAE"], y=model_names,
        orientation="h",
        marker=dict(color=[c.replace(")", ",0.7)").replace("rgb","rgba") if "rgb" in c else c for c in bar_colors], cornerradius=4),
        text=[f"{v:.4f}" for v in results_df["MAE"]], textposition="auto",
        textfont=dict(color="#f1f5f9", size=11),
        hovertemplate="<b>%{y}</b><br>MAE: %{x:.4f} °C<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        name="RMSE", x=results_df["RMSE"], y=model_names,
        orientation="h",
        marker=dict(color="rgba(148,163,184,0.25)", cornerradius=4),
        text=[f"{v:.4f}" for v in results_df["RMSE"]], textposition="auto",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>RMSE: %{x:.4f} °C<extra></extra>",
    ))
    fig_bar.update_layout(**chart_layout(
        title=dict(text="MAE & RMSE by Model (lower is better)"), height=320,
        barmode="group",
        xaxis=dict(title="Error (°C)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=110, r=24, t=44, b=44),
    ))
    st.plotly_chart(fig_bar, width="stretch")

# ── Actual vs Predicted ───────────────────────────────────────────────────────
section_label("Actual vs Predicted — Last CV Fold")

fig_avp = go.Figure()
# Actual (shared across all models)
any_fold = list(last_fold_data.values())[0]
x_actual = any_fold.get("years") if any_fold.get("years") is not None else list(range(len(any_fold["y_test"])))
fig_avp.add_trace(go.Scatter(
    x=x_actual, y=any_fold["y_test"],
    mode="lines+markers", name="Actual",
    line=dict(color="#f1f5f9", width=2.5, dash="dot"),
    marker=dict(size=7, color="#f1f5f9", symbol="diamond"),
    hovertemplate="<b>Actual</b><br>Year: %{x}<br>Value: %{y:.4f} °C<extra></extra>",
))
for model_name, fold in last_fold_data.items():
    x_ax = fold.get("years") if fold.get("years") is not None else list(range(len(fold["y_test"])))
    color = MODEL_COLORS.get(model_name, DS["cyan"])
    fig_avp.add_trace(go.Scatter(
        x=x_ax, y=fold["y_pred"],
        mode="lines+markers", name=model_name,
        line=dict(color=color, width=2),
        marker=dict(size=5),
        hovertemplate=f"<b>{model_name}</b><br>Year: %{{x}}<br>Predicted: %{{y:.4f}} °C<extra></extra>",
    ))
fig_avp.update_layout(**chart_layout(
    title=dict(text="Actual vs Predicted Temperature Anomaly (Last Cross-Validation Fold)"),
    height=400,
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(title="Temperature Anomaly (°C)", gridcolor="rgba(255,255,255,0.05)"),
))
st.plotly_chart(fig_avp, width="stretch")

# ── Feature Importance ────────────────────────────────────────────────────────
section_label("Feature Importance — XGBoost")

col_fi, col_shap = st.columns(2)

model = get_xgb_model(X, y)
fi_df = get_feature_importance(model, features)

with col_fi:
    if fi_df is not None and not fi_df.empty:
        fi_sorted = fi_df.sort_values("Importance", ascending=True)
        max_imp = fi_sorted["Importance"].max()
        fi_colors = [f"rgba(34,211,238,{0.3 + 0.7 * (v / max_imp) if max_imp > 0 else 0.7})" for v in fi_sorted["Importance"]]

        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            x=fi_sorted["Importance"], y=fi_sorted["Feature"],
            orientation="h",
            marker=dict(color=fi_colors, cornerradius=4),
            text=[f"{v:.4f}" for v in fi_sorted["Importance"]],
            textposition="auto", textfont=dict(color="#f1f5f9", size=11),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(**chart_layout(
            title=dict(text="XGBoost Feature Importance"), height=340,
            showlegend=False,
            xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=110, r=24, t=44, b=44),
        ))
        st.plotly_chart(fig_fi, width="stretch")

with col_shap:
    with st.spinner("Computing SHAP values…"):
        shap_vals, _ = compute_shap_values(model, X)
    shap_df = shap_summary_df(shap_vals, features)

    if not shap_df.empty:
        shap_sorted = shap_df.sort_values("Mean |SHAP|", ascending=True)
        max_shap = shap_sorted["Mean |SHAP|"].max()
        shap_colors = [f"rgba(244,63,94,{0.3 + 0.7 * (v / max_shap) if max_shap > 0 else 0.7})" for v in shap_sorted["Mean |SHAP|"]]

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=shap_sorted["Mean |SHAP|"], y=shap_sorted["Feature"],
            orientation="h",
            marker=dict(color=shap_colors, cornerradius=4),
            text=[f"{v:.4f}" for v in shap_sorted["Mean |SHAP|"]],
            textposition="auto", textfont=dict(color="#f1f5f9", size=11),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        fig_shap.update_layout(**chart_layout(
            title=dict(text="SHAP Feature Importance"), height=340,
            showlegend=False,
            xaxis=dict(title="Mean |SHAP| Value", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=110, r=24, t=44, b=44),
        ))
        st.plotly_chart(fig_shap, width="stretch")

# ── What drives the forecast ──────────────────────────────────────────────────
section_label("What Drives the Forecast?")

if not shap_df.empty:
    top_feature  = shap_df.iloc[0]["Feature"]
    top_shap_val = shap_df.iloc[0]["Mean |SHAP|"]
    second_feat  = shap_df.iloc[1]["Feature"] if len(shap_df) > 1 else ""

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(insight_card(
            f"<b>{top_feature}</b> is the single most influential predictor, contributing an average of "
            f"<b>{top_shap_val:.4f} °C</b> of SHAP impact per prediction. "
            f"This aligns with the physical understanding that cumulative emissions drive long-term warming.",
            "cyan",
        ), unsafe_allow_html=True)
    with c2:
        if second_feat:
            second_shap = shap_df[shap_df["Feature"] == second_feat]["Mean |SHAP|"].values[0]
            st.markdown(insight_card(
                f"<b>{second_feat}</b> is the second most important feature (mean |SHAP| = {second_shap:.4f}). "
                f"Together, the top two features account for "
                f"<b>{(top_shap_val + second_shap) / shap_df['Mean |SHAP|'].sum() * 100:.1f}%</b> of total model impact.",
                "amber",
            ), unsafe_allow_html=True)

# ── Methodology ───────────────────────────────────────────────────────────────
with st.expander("Methodology — TimeSeriesSplit Cross-Validation"):
    st.markdown("""
**TimeSeriesSplit** (expanding window) ensures no future data leaks into training:

- Split 1: Train on ~17% → Test on next ~17%
- Split 5 (last): Train on ~83% → Test on final ~17%

**Models evaluated:** Linear Regression · Random Forest (200 trees) · XGBoost (200 estimators, lr=0.05, depth=4) · ARIMA(1,1,1) with exogenous regressors

**Features:** `year`, `co2_ppm`, `cumulative_co2`, `co2_emissions`, `co2_lag_1`, `co2_lag_5`

**Note:** Negative R² in time-series CV is common when the model fails to extrapolate a strong upward trend better than a horizontal mean baseline.
    """)

with st.expander("Raw cross-validation results"):
    st.dataframe(results_df.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R2": "{:.6f}"}), width="stretch")
