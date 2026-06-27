"""
FPL Squad Optimizer - Streamlit Dashboard
Run via: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config FIRST
st.set_page_config(
    page_title="FPL Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Core imports
from src.ingestion import load_all_data
from src.metrics import engineer_features, TeamDynamicsAnalyzer
from src.optimizer import OptimizerConfig, optimize_squad, pick_starting_xi, VALID_FORMATIONS

# ── Theme Toggle ─────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# ── CSS Design System ─────────────────────────────────────────────────────────
CSS = f"""
<style>
:root {{
    --bg: {'#09090b' if IS_DARK else '#ffffff'};
    --bg-subtle: {'#0c0c0f' if IS_DARK else '#f9fafb'};
    --card: {'#0c0c0f' if IS_DARK else '#ffffff'};
    --card-hover: {'#131316' if IS_DARK else '#f4f4f5'};
    --border: {'#1e1e24' if IS_DARK else '#e4e4e7'};
    --border-subtle: {'#16161a' if IS_DARK else '#f0f0f2'};
    --text: {'#fafafa' if IS_DARK else '#09090b'};
    --text-muted: #71717a;
    --text-dim: {'#52525b' if IS_DARK else '#a1a1aa'};
    --accent: #2563eb;
    --accent-muted: #1d4ed8;
    --green: {'#22c55e' if IS_DARK else '#16a34a'};
    --green-muted: {'rgba(34,197,94,0.12)' if IS_DARK else 'rgba(22,163,74,0.08)'};
    --red: {'#ef4444' if IS_DARK else '#dc2626'};
    --red-muted: {'rgba(239,68,68,0.12)' if IS_DARK else 'rgba(220,38,38,0.08)'};
    --amber: {'#f59e0b' if IS_DARK else '#d97706'};
    --amber-muted: {'rgba(245,158,11,0.12)' if IS_DARK else 'rgba(217,119,6,0.08)'};
    --shadow: {'none' if IS_DARK else '0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)'};
    --radius: 10px;
}}

/* Hide Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton {{
    display: none !important;
}}

/* Global App Styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.835rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px;
}}

/* Metrics */
.metric-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.4rem; box-shadow: var(--shadow); }}
.metric-label {{ font-size: 0.85rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }}
.metric-value {{ font-size: 2.0rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em; margin-top: 0.2rem; }}

/* Charts */
.chart-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem 1.2rem 0.6rem; box-shadow: var(--shadow); margin-top: 1rem; }}
.chart-title {{ font-size: 0.9rem; font-weight: 600; color: var(--text); }}
.chart-subtitle {{ font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.8rem; }}

/* Tables */
.data-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.85rem; margin-top: 1rem; }}
.data-table th {{ text-align: left; padding: 0.8rem 1rem; color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 2px solid var(--border); }}
.data-table td {{ padding: 0.75rem 1rem; color: var(--text); border-bottom: 1px solid var(--border-subtle); vertical-align: middle; }}
.data-table tr:last-child td {{ border-bottom: none; }}

/* Badges */
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }}
.badge-GK {{ color: var(--amber); background: var(--amber-muted); }}
.badge-DEF {{ color: var(--green); background: var(--green-muted); }}
.badge-MID {{ color: var(--accent); background: rgba(37,99,235,0.1); }}
.badge-FWD {{ color: var(--red); background: var(--red-muted); }}

.brand-name {{ font-size: 1.5rem; font-weight: 800; color: var(--text); }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#a1a1aa" if IS_DARK else "#71717a", size=12),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
    ),
)

def render_table(df, title=""):
    rows = ""
    for _, r in df.iterrows():
        pos = r['position']
        name = r.get('web_name', 'Unknown')
        team = r.get('team_name', '')
        cost = f"£{r['now_cost']:.1f}m"
        pts = f"{r.get('projected_points', 0):.2f}"
        val = f"{r.get('value_score', 0):.3f}"
        
        rows += f"""<tr>
            <td><span class="badge badge-{pos}">{pos}</span></td>
            <td style="font-weight: 600;">{name}</td>
            <td>{team}</td>
            <td>{cost}</td>
            <td style="font-weight: 600; color: var(--green);">{pts}</td>
            <td>{val}</td>
        </tr>"""
        
    st.markdown(f"""
    <div class="chart-wrap">
        <div class="chart-title">{title}</div>
        <table class="data-table">
            <thead><tr><th>Pos</th><th>Player</th><th>Club</th><th>Cost</th><th>Proj Pts</th><th>Value Score</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ── App Layout ───────────────────────────────────────────────────────────────

head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown('<div class="brand"><span class="brand-name">⚽ FPL Squad Optimizer</span></div>', unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    budget = st.slider("Total Budget (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)
    formation = st.selectbox("Formation", list(VALID_FORMATIONS.keys()), index=3) # Default 4-4-2
    max_club = st.slider("Max Players per Club", min_value=1, max_value=5, value=3)
    gws = st.slider("Upcoming GWs for FDR", min_value=1, max_value=5, value=1)
    
    st.markdown("---")
    force_refresh = st.checkbox("Force API Refresh", value=False)
    run_btn = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)


# ── Core Logic ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_data(refresh):
    return load_all_data(force_refresh=refresh)

@st.cache_data(ttl=3600)
def get_features(_players, _fixtures, gws):
    return engineer_features(_players, _fixtures, next_n_gws=gws)

try:
    with st.spinner("Fetching FPL API Data..."):
        players_df, fixtures_df = get_data(force_refresh)
    
    enriched_df = get_features(players_df, fixtures_df, gws)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


if run_btn or "optim_result" not in st.session_state:
    with st.spinner("Solving Knapsack ILP..."):
        config = OptimizerConfig(
            total_budget=budget,
            max_players_per_club=max_club,
            objective_weight="projected_points"
        )
        res = optimize_squad(enriched_df, config=config)
        st.session_state.optim_result = res


res = st.session_state.optim_result

if not res.is_optimal:
    st.error(f"### ❌ Optimization Failed\n**Reason:** {res.infeasibility_reason}")
else:
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Cost", f"£{res.total_cost:.1f}m")
    with c2: metric_card("Projected Points", f"{res.total_projected_points:.1f} pts")
    with c3: metric_card("Avg Value Score", f"{res.squad['value_score'].mean():.2f}")
    with c4: metric_card("Available Funds", f"£{budget - res.total_cost:.1f}m")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Optimal Squad", "📊 Value Analysis", "🏟️ Team Dynamics"])
    
    with tab1:
        try:
            starting_xi, bench = pick_starting_xi(res.squad, formation=formation)
            
            c_left, c_right = st.columns([6, 4])
            with c_left:
                render_table(starting_xi, f"Starting XI [{formation}]")
            with c_right:
                render_table(bench, "Bench")
                
        except Exception as e:
            st.error(f"Formation Error: {e}")
            render_table(res.squad, "Full 15-Man Squad")

    with tab2:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Cost vs Projected Points</div>
            <div class="chart-subtitle">Identify high-value players in your optimal squad</div>
        """, unsafe_allow_html=True)
        
        fig = px.scatter(
            res.squad, x="now_cost", y="projected_points", 
            color="position", text="web_name", size="value_score",
            color_discrete_map={"GK": "#f59e0b", "DEF": "#22c55e", "MID": "#3b82f6", "FWD": "#ef4444"},
            labels={"now_cost": "Cost (£m)", "projected_points": "Projected Pts"}
        )
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        analyzer = TeamDynamicsAnalyzer(enriched_df)
        report = analyzer.systemic_impact_report().head(12)
        
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Club Systemic Impact</div>
            <div class="chart-subtitle">Average Projected Points by Club (Attack vs Defense)</div>
        """, unsafe_allow_html=True)
        
        fig2 = go.Figure(data=[
            go.Bar(name='Attack (MID/FWD)', x=report['team_name'], y=report['avg_attack_pts'], marker_color='#ef4444'),
            go.Bar(name='Defense (GK/DEF)', x=report['team_name'], y=report['avg_defense_pts'], marker_color='#22c55e')
        ])
        fig2.update_layout(
            barmode='group',
            **PLOT_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("</div>", unsafe_allow_html=True)
