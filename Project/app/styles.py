"""
app/styles.py
Shared design system for the Climate Intelligence Platform.
Import this in every page:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from styles import DS, apply_global_css, page_header, kpi_row, insight_card, section_label, chart_layout, add_thresholds
"""
import streamlit as st
import plotly.graph_objects as go

# ── Design tokens ─────────────────────────────────────────────────────────────
DS = {
    # Backgrounds
    "bg":          "#0a0f1a",
    "surface":     "#0f1923",
    "surface2":    "#152030",
    "border":      "rgba(255,255,255,0.07)",
    "border_accent":"rgba(34,211,238,0.18)",
    # Accents
    "cyan":        "#22d3ee",
    "cyan_dim":    "rgba(34,211,238,0.10)",
    "coral":       "#f43f5e",
    "coral_dim":   "rgba(244,63,94,0.10)",
    "amber":       "#f59e0b",
    "amber_dim":   "rgba(245,158,11,0.10)",
    "green":       "#10b981",
    "green_dim":   "rgba(16,185,129,0.10)",
    "violet":      "#a78bfa",
    "violet_dim":  "rgba(167,139,250,0.10)",
    # Typography
    "text":        "#f1f5f9",
    "text_muted":  "#94a3b8",
    "text_dim":    "#475569",
    # Radius
    "radius":      "8px",
    "radius_lg":   "12px",
}

# Ordered palette for multi-series charts
SERIES_COLORS = ["#22d3ee", "#f43f5e", "#f59e0b", "#10b981", "#a78bfa", "#fb923c"]


# ── Global CSS ────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0f1a;
    color: #f1f5f9;
}

/* Reduce default Streamlit top padding */
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(100deg, #0f2333 0%, #102a3a 60%, #0a1f2e 100%);
    border-left: 3px solid #22d3ee;
    border-radius: 8px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.4rem;
}
.page-header h1 {
    font-size: 1.45rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 0.15rem 0; letter-spacing: -0.3px;
}
.page-header p {
    font-size: 0.85rem; color: #94a3b8; margin: 0;
}

/* ── KPI card ── */
.kpi-card {
    background: #0f1923;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    text-align: left;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(34,211,238,0.35); }
.kpi-label {
    font-size: 0.72rem; font-weight: 500; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 0.35rem;
}
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.kpi-delta { font-size: 0.78rem; margin-top: 0.25rem; color: #94a3b8; }
.kpi-delta.up   { color: #f43f5e; }
.kpi-delta.down { color: #10b981; }
.kpi-delta.neutral { color: #f59e0b; }

/* ── Insight card ── */
.insight-card {
    background: #0f1923;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.insight-card.cyan   { border-left: 3px solid #22d3ee; }
.insight-card.coral  { border-left: 3px solid #f43f5e; }
.insight-card.amber  { border-left: 3px solid #f59e0b; }
.insight-card.green  { border-left: 3px solid #10b981; }
.insight-card.violet { border-left: 3px solid #a78bfa; }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 1.1px;
    margin: 1.6rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Data quality badge ── */
.badge {
    display: inline-block;
    font-size: 0.7rem; font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    margin-right: 0.4rem;
    letter-spacing: 0.4px;
}
.badge.green  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.badge.amber  { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.badge.red    { background: rgba(244,63,94,0.15);   color: #f43f5e; border: 1px solid rgba(244,63,94,0.25); }
.badge.neutral{ background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.20); }

/* ── Explore tile ── */
.explore-tile {
    background: #0f1923;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    text-decoration: none;
}
.explore-tile:hover { border-color: rgba(34,211,238,0.4); background: #102130; }
.explore-tile-title { font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.15rem; }
.explore-tile-desc  { font-size: 0.78rem; color: #94a3b8; }

/* ── Warning box ── */
.warning-box {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem; color: #fbbf24;
    margin-top: 0.8rem;
}

/* ── Streamlit metric overrides ── */
div[data-testid="stMetric"] {
    background: #0f1923 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    padding: 0.9rem 1.1rem !important;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.8px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.65rem !important; font-weight: 700; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #080d15 !important; }
[data-testid="stSidebar"] .sidebar-content { padding-top: 1rem; }
"""


def apply_global_css():
    """Call once per page, after set_page_config."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ── Component helpers ─────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = ""):
    sub_html = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str = "", direction: str = "neutral") -> str:
    """Return HTML string for a KPI card. Use in st.markdown(..., unsafe_allow_html=True)."""
    arrow = {"up": "▲", "down": "▼", "neutral": "—"}.get(direction, "—")
    delta_html = (
        f'<div class="kpi-delta {direction}">{arrow} {delta}</div>'
        if delta else ""
    )
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def insight_card(text: str, accent: str = "cyan") -> str:
    """Return HTML for an insight card. accent: cyan | coral | amber | green | violet."""
    return f'<div class="insight-card {accent}">{text}</div>'


def section_label(text: str):
    """Render a small uppercase section divider label."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = "neutral") -> str:
    """Return HTML for an inline badge. color: green | amber | red | neutral."""
    return f'<span class="badge {color}">{text}</span>'


def warning_box(text: str):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def chart_layout(**overrides) -> dict:
    """Base Plotly layout. Pass keyword overrides to customise."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,26,0.6)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=48, r=24, t=44, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#152030", font_size=12, font_family="Inter"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, linecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, linecolor="rgba(255,255,255,0.08)"),
        title=dict(font=dict(size=14, color="#f1f5f9"), x=0, xanchor="left", pad=dict(l=0)),
    )
    base.update(overrides)
    return base


def add_thresholds(fig: go.Figure, thresholds=(1.5, 2.0), colors=("#f59e0b", "#f43f5e")):
    """Add horizontal 1.5°C and 2.0°C dashed threshold lines to a figure."""
    labels = ["1.5°C target", "2.0°C limit"]
    for val, color, label in zip(thresholds, colors, labels):
        fig.add_hline(
            y=val,
            line=dict(color=color, width=1.2, dash="dash"),
            annotation=dict(
                text=label, font=dict(size=10, color=color),
                xanchor="right", x=1,
            ),
        )
    return fig
