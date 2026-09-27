"""
================================================================================
  Autonomous Micro-Grid Governor  (AMG-2056)
  Decentralized Feeder Intermittency & Hamiltonian Neural Network Management
================================================================================
  Stack  :  Python · Streamlit · Plotly · NumPy · Pandas
  Purpose:  Simulates a neighborhood sub-distributor / transformer secondary
            managing rooftop solar, community battery storage, and dynamic
            local load profiles with real-time PINN / HNN edge inference.
================================================================================
"""

import datetime
import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMG-2056 · Autonomous Micro-Grid Governor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  SESSION STATE DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "tick": 0,
    "bess_soc": 74.2,
    "feeder_voltage": 230.4,
    "solar_gen": 19.35,
    "current_load": 32.80,
    "h_loss": 0.0012,
    "radio_connected": True,
    "manual_override": False,
    "weather": "Normal",
    "pinn_epoch": 140,
    "auto_refresh": True,
    "refresh_interval": 3,
    "dark_mode": True,
    "log_entries": [
        ("HNN",      "19:40:01", "AMG-2056 controller online. Local grid telemetry synced."),
        ("HNN",      "19:40:05", "Energy invariant verified (dH/dt = 0.0008 pu). System stable."),
        ("DISPATCH", "19:40:10", "Rooftop solar PV input 19.2 kW detected at transformer secondary."),
        ("DISPATCH", "19:40:15", "Community BESS operating in autonomous droop voltage regulation mode."),
    ],
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

DARK = st.session_state["dark_mode"]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  DESIGN TOKENS  (dark / light adaptive)
# ──────────────────────────────────────────────────────────────────────────────
if DARK:
    T = {
        # Deep navy — familiar night-shift SCADA background
        "bg_page":      "#0F1C2E",
        "bg_card":      "#162438",
        "bg_inner":     "#0B1624",
        "bg_terminal":  "#080F1C",
        "border":       "#1E3349",
        "border_hi":    "#2B4468",
        # Warm off-white text — easy on eyes during long shifts
        "text_primary": "#D8E4F0",
        "text_sec":     "#A8BDD4",
        "text_muted":   "#5E7A96",
        "plot_bg":      "#0F1C2E",
        "grid":         "#1A2E44",
        "axis":         "#2B4468",
        "shadow":       "0 4px 20px rgba(0,0,0,0.45)",
        "shadow_card":  "0 2px 10px rgba(0,0,0,0.30)",
        "sidebar_bg":   "#0B1624",
    }
else:
    T = {
        # Clean light mode — like a printed engineering report
        "bg_page":      "#EBF0F7",
        "bg_card":      "#FFFFFF",
        "bg_inner":     "#F4F8FC",
        "bg_terminal":  "#EEF4FB",
        "border":       "#C0D0E0",
        "border_hi":    "#7A9AB8",
        "text_primary": "#1A2B3C",
        "text_sec":     "#2C4A64",
        "text_muted":   "#5E7A96",
        "plot_bg":      "#FFFFFF",
        "grid":         "#D8E6F0",
        "axis":         "#7A9AB8",
        "shadow":       "0 2px 10px rgba(0,0,0,0.09)",
        "shadow_card":  "0 1px 5px rgba(0,0,0,0.06)",
        "sidebar_bg":   "#DDE8F2",
    }

# ── Status colours: standard industrial RAG + corporate blue ──
# Selected to match industry-standard HMI/SCADA palette used in
# feeder automation and distribution management systems:
#   Green  = safe operation / nominal      (traffic-light green)
#   Orange = caution / warning condition   (traffic-light orange)
#   Red    = alarm / critical fault        (universal alarm red)
#   Blue   = information / telemetry link  (corporate info blue)
#   Purple = neural network / HNN metrics  (convention in academia)
C = {
    "emerald": "#28A745",   # safe-green  (replaces neon #10B981)
    "cyan":    "#0D6EFD",   # info-blue   (replaces cyan #06B6D4)
    "amber":   "#FD7E14",   # warm orange (replaces cold amber #F59E0B)
    "rose":    "#DC3545",   # alarm-red   (replaces rose #F43F5E)
    "indigo":  "#6F42C1",   # neural network / HNN convergence metrics
    "violet":  "#6F42C1",
}

# ──────────────────────────────────────────────────────────────────────────────
# 4.  GLOBAL CSS  (injected once per render)
# ──────────────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&family=Roboto+Mono:wght@400;500;700&display=swap');

:root {{
    --bg-page:    {T["bg_page"]};
    --bg-card:    {T["bg_card"]};
    --bg-inner:   {T["bg_inner"]};
    --border:     {T["border"]};
    --border-hi:  {T["border_hi"]};
    --tp:         {T["text_primary"]};
    --ts:         {T["text_sec"]};
    --tm:         {T["text_muted"]};
    --green:  #28A745; --blue:  #0D6EFD;
    --orange: #FD7E14; --red:   #DC3545; --purple: #6F42C1;
}}

html, body, [class*="css"], .stApp {{
    font-family: 'Roboto', 'Segoe UI', Arial, sans-serif !important;
    background-color: var(--bg-page) !important;
    color: var(--tp) !important;
    font-size: 14px;
    letter-spacing: 0.01em;
}}

.block-container {{
    padding: 1rem 2rem 2.5rem 2rem !important;
    max-width: 1440px !important;
}}

/* ── Equal-height rows ── */
[data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
[data-testid="stColumn"] {{ display: flex !important; flex-direction: column !important; }}
[data-testid="stColumn"] > div {{
    display: flex !important; flex-direction: column !important; flex: 1 1 auto !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    height: 100% !important;
    display: flex !important; flex-direction: column !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: {T["shadow"]} !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    display: flex !important; flex-direction: column !important; flex: 1 1 auto !important;
}}
[data-testid="stVerticalBlock"] {{
    flex: 1 1 auto !important; display: flex !important; flex-direction: column !important;
}}

/* ── Page Header ── */
.amg-hdr {{
    background: var(--bg-card);
    border: 1px solid var(--border-hi);
    border-radius: 14px;
    padding: 16px 22px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 12px; margin-bottom: 18px;
    box-shadow: {T["shadow"]};
}}
.amg-icon {{
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #1565C0 0%, #0D6EFD 100%);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: #fff;
    box-shadow: 0 0 18px rgba(13,110,253,0.30); flex-shrink:0;
}}
.amg-title {{ font-size:1.25rem; font-weight:700; color:var(--tp); line-height:1.25; letter-spacing:0.01em; }}
.amg-sub   {{ font-size:0.75rem; color:var(--tm); font-weight:400; margin-top:3px; letter-spacing:0.02em; }}

/* ── Status pills — standard RAG ── */
.pill {{
    display:inline-flex; align-items:center; gap:6px;
    font-size:0.72rem; font-weight:600; padding:4px 12px;
    border-radius:4px; white-space:nowrap; line-height:1;
    letter-spacing:0.03em;
}}
.pill-g {{ background:rgba(40,167,69,.14);  color:#28A745; border:1px solid rgba(40,167,69,.40); }}
.pill-c {{ background:rgba(13,110,253,.14); color:#0D6EFD; border:1px solid rgba(13,110,253,.40); }}
.pill-a {{ background:rgba(253,126,20,.14); color:#FD7E14; border:1px solid rgba(253,126,20,.40); }}
.pill-r {{ background:rgba(220,53,69,.15);  color:#DC3545; border:1px solid rgba(220,53,69,.40); }}
.dot   {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.dg {{ background:#28A745; box-shadow:0 0 5px #28A745; }}
.dc {{ background:#0D6EFD; box-shadow:0 0 5px #0D6EFD; }}
.da {{ background:#FD7E14; box-shadow:0 0 5px #FD7E14; }}
.dr {{ background:#DC3545; box-shadow:0 0 5px #DC3545; }}

/* ── KPI Cards ── */
.kpi {{
    height:142px;
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:8px; padding:16px 20px;
    display:flex; flex-direction:column; justify-content:space-between;
    box-shadow:{T["shadow_card"]};
    transition:border-color .18s ease, box-shadow .18s ease;
    box-sizing:border-box; overflow:hidden; position:relative;
}}
.kpi:hover {{ border-color:var(--border-hi); box-shadow:{T["shadow"]}; }}
.kpi-bar {{ position:absolute; top:0; left:0; width:4px; height:100%; border-radius:8px 0 0 8px; }}
.kpi-lbl {{
    font-size:0.70rem; font-weight:500; letter-spacing:0.10em;
    text-transform:uppercase; color:var(--tm); font-family:'Roboto',sans-serif;
}}
.kpi-val {{
    font-family:'Roboto Mono',monospace;
    font-size:1.85rem; font-weight:700; line-height:1.05; color:var(--tp);
}}
.kpi-val span {{ font-size:0.95rem; color:var(--tm); font-weight:400; margin-left:2px; }}
.kpi-foot {{ display:flex; align-items:center; justify-content:space-between; font-size:0.70rem; color:var(--ts); }}
.kdelta {{
    display:inline-flex; align-items:center; gap:3px;
    padding:2px 8px; border-radius:3px; font-weight:600;
    font-size:0.68rem; font-family:'Roboto Mono',monospace;
}}
.kdg {{ background:rgba(40,167,69,.14);  color:#28A745; border:1px solid rgba(40,167,69,.4); }}
.kda {{ background:rgba(253,126,20,.14); color:#FD7E14; border:1px solid rgba(253,126,20,.4); }}
.kdc {{ background:rgba(13,110,253,.14); color:#0D6EFD; border:1px solid rgba(13,110,253,.4); }}
.kdr {{ background:rgba(220,53,69,.14);  color:#DC3545; border:1px solid rgba(220,53,69,.4); }}

/* ── Panel section header ── */
.sec-hdr {{
    display:flex; justify-content:space-between; align-items:center;
    padding:13px 16px 10px 16px;
    border-bottom:1px solid var(--border); flex-shrink:0;
}}
.sec-ttl {{ font-size:0.84rem; font-weight:700; color:var(--tp); display:flex; align-items:center; gap:7px; }}
.sec-sub {{ font-size:0.69rem; color:var(--tm); font-weight:500; }}

/* ── Terminal / Event log  (fixed-height, scrollable) ── */
.term {{
    height: 320px;
    overflow-y: scroll;
    font-family: 'Roboto Mono', monospace;
    background:{T["bg_terminal"]}; border-radius:8px;
    margin:12px 14px 12px 14px; padding:10px 12px;
    font-size:0.70rem; line-height:1.85;
    border:1px solid var(--border-hi);
    scrollbar-width: thin;
    scrollbar-color: #0D6EFD transparent;
}}
.term::-webkit-scrollbar {{ width: 6px; }}
.term::-webkit-scrollbar-track {{ background: transparent; border-radius:4px; }}
.term::-webkit-scrollbar-thumb {{ background: #0D6EFD; border-radius:4px; }}
.term::-webkit-scrollbar-thumb:hover {{ background: #3D8BFD; }}
.tl {{
    display:flex; align-items:flex-start; gap:8px;
    padding:3px 4px; border-radius:4px; margin-bottom:4px;
}}
.tl:hover {{ background:rgba({"255,255,255" if DARK else "0,0,0"},0.04); }}
.tts  {{ color:var(--tm); white-space:nowrap; flex-shrink:0; font-size:0.62rem; min-width:65px; }}
.ttag {{ font-weight:700; font-size:0.60rem; padding:2px 7px; border-radius:3px; white-space:nowrap; flex-shrink:0; min-width:60px; text-align:center; }}
.t-hnn  {{ background:rgba(6,182,212,.18);  color:#06B6D4; border:1px solid rgba(6,182,212,.4); }}
.t-dsp  {{ background:rgba(16,185,129,.18); color:#10B981; border:1px solid rgba(16,185,129,.4); }}
.t-wrn  {{ background:rgba(245,158,11,.18); color:#F59E0B; border:1px solid rgba(245,158,11,.4); }}
.t-ovr  {{ background:rgba(244,63,94,.18);  color:#F43F5E; border:1px solid rgba(244,63,94,.4); }}
.tmsg {{ color:var(--ts); word-break:break-word; flex:1; }}

/* ── BESS Diagnostic grid ── */
.bess-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 0 14px 14px 14px;
}}
.bess-tile {{
    background: {T["bg_inner"]};
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}}
.bess-tile-lbl {{
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--tm);
    font-family: 'Roboto', sans-serif;
}}
.bess-tile-val {{
    font-family: 'Roboto Mono', monospace;
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--tp);
    line-height: 1.2;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color:{T["sidebar_bg"]} !important;
    border-right:1px solid var(--border) !important;
}}
.sb-brand {{ text-align:center; padding:4px 0 16px 0; border-bottom:1px solid var(--border); margin-bottom:16px; }}
.sb-badge {{ display:inline-block; background:#0D6EFD; color:#FFFFFF; font-weight:700; font-size:0.76rem; padding:4px 12px; border-radius:4px; letter-spacing:0.06em; margin-bottom:6px; font-family:'Roboto',sans-serif; }}
.sb-ttl {{ font-size:0.95rem; font-weight:700; color:var(--tp); font-family:'Roboto',sans-serif; }}
.sb-sub {{ font-size:0.67rem; color:var(--tm); text-transform:uppercase; letter-spacing:0.08em; font-weight:400; }}
.sb-sec {{ font-size:0.68rem; font-weight:700; color:#0D6EFD; text-transform:uppercase; letter-spacing:0.10em; margin:16px 0 7px 0; font-family:'Roboto',sans-serif; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--border-hi); border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:#0D6EFD; }}

.gap14 {{ height:14px; }} .gap10 {{ height:10px; }}

/* ═══════════════════════════════════════════════════════════════════════
   PER-THEME STREAMLIT WIDGET OVERRIDES
   Forces ALL Streamlit-generated text (labels, radio, toggle, slider,
   markdown paragraphs, expanders) to the correct foreground colour.
   config.toml uses base="light"; dark mode CSS takes over below.
═══════════════════════════════════════════════════════════════════════ */

/* ── Streamlit app canvas & page background ── */
.stApp, [data-testid="stAppViewContainer"] {{
    background-color: {T["bg_page"]} !important;
}}
[data-testid="stHeader"] {{
    background-color: {T["bg_page"]} !important;
}}

/* ── Every native text node in the page ── */
.stApp p, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp li, .stApp small, .stApp strong,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stExpanderToggleIcon"],
.stRadio label p, .stRadio label span,
.stCheckbox label p, .stCheckbox label span,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
.stSlider p, .stSlider span,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
.streamlit-expanderHeader,
.streamlit-expanderContent p,
[data-testid="stSelectbox"] p,
[data-baseweb="select"] span {{
    color: {T["text_primary"]} !important;
}}

/* ── Muted / helper text ── */
.stApp small,
[data-testid="stCaptionContainer"] p {{
    color: {T["text_muted"]} !important;
}}

/* ── Secondary text (sub-labels) ── */
[data-testid="stWidgetLabel"] small {{
    color: {T["text_sec"]} !important;
}}

/* ── Sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {T["text_primary"]} !important;
}}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: {T["text_muted"]} !important;
}}

/* ── Input & select backgrounds ── */
[data-baseweb="input"], [data-baseweb="select"],
[data-baseweb="textarea"] {{
    background-color: {T["bg_inner"]} !important;
    color: {T["text_primary"]} !important;
    border-color: {T["border"]} !important;
}}

.gap14 {{ height:14px; }} .gap10 {{ height:10px; }}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PHYSICS / NEURAL SIMULATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def gen_24h_data(weather: str) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    h   = np.linspace(0, 24, 288)
    load = 11.5 + 13.5 * np.exp(-((h - 8.2)**2)/4.5) + 23.0 * np.exp(-((h - 19.8)**2)/5.2)
    if weather == "Special Function/Ceremony Unexpected Load Spike":
        load += 28.5 * np.exp(-((h - 19.5)**2)/1.8)
    act_load  = np.clip(load + rng.normal(0, 0.42, 288), 6.0, 78.0)
    pred_load = act_load + rng.normal(0, 0.55, 288) - 0.25 * np.sin(h)
    solar = np.clip(33.5 * np.exp(-((h - 12.8)**2)/8.8), 0, None)
    if weather == "Sudden Cloud Cover":
        solar *= 1.0 - 0.82 * np.exp(-((h - 12.4)**2)/1.1)
    elif weather == "Special Function/Ceremony Unexpected Load Spike":
        solar *= 0.90
    act_solar  = np.clip(solar + rng.normal(0, 0.35, 288), 0.0, 38.0)
    pred_solar = np.clip(act_solar + rng.normal(0, 0.48, 288) + 0.35*np.exp(-((h-11.8)**2)/9), 0.0, 38.0)
    today = datetime.datetime.now().date()
    ts = [datetime.datetime.combine(today, datetime.time(0,0)) + datetime.timedelta(minutes=5*i) for i in range(288)]
    return pd.DataFrame({"Time": ts, "Actual Load": act_load, "HNN Predicted Load": pred_load,
                         "Actual Solar": act_solar, "HNN Predicted Solar": pred_solar})


def gen_loss_data(_epoch: int) -> pd.DataFrame:
    rng    = np.random.default_rng(77)
    epochs = np.arange(1, 161)
    total  = np.clip(0.88*np.exp(-epochs/30) + 0.012 + rng.normal(0, 0.0018, 160), 0.010, 1.2)
    h_err  = np.clip(0.45*np.exp(-epochs/36) + 0.0012+ rng.normal(0, 0.0004, 160), 0.0009,0.6)
    mse    = np.clip(0.55*np.exp(-epochs/26) + 0.008 + rng.normal(0, 0.0010, 160), 0.006, 0.8)
    return pd.DataFrame({"Epoch": epochs, "Total": total, "H_error": h_err, "MSE": mse})


def sim_tick(weather: str, override: bool, radio: bool) -> dict:
    now    = datetime.datetime.now()
    ts_str = now.strftime("%H:%M:%S")
    hf     = now.hour + now.minute / 60.0
    pv = 32.0 * max(0.0, float(np.exp(-((hf-12.8)**2)/8.8)))
    if weather == "Sudden Cloud Cover":
        pv *= random.uniform(0.15, 0.35)
    elif weather == "Special Function/Ceremony Unexpected Load Spike":
        pv *= random.uniform(0.85, 0.95)
    else:
        pv *= random.uniform(0.95, 1.05)
    solar = round(max(0.0, pv + random.gauss(0, 0.3)), 2)
    load  = 11.5 + 13.5*max(0.0,float(np.exp(-((hf-8.2)**2)/4.5))) \
                 + 23.0*max(0.0,float(np.exp(-((hf-19.8)**2)/5.2)))
    if weather == "Special Function/Ceremony Unexpected Load Spike":
        load += random.uniform(22.0, 31.0)
    if override:
        load *= 0.65
    load = round(max(5.0, load + random.gauss(0, 0.5)), 2)
    net  = solar - load
    soc_step = -random.uniform(0.4, 0.7) if override else (net/38.0)*random.uniform(0.85,1.15)
    bess = round(float(np.clip(st.session_state["bess_soc"] + soc_step, 12.0, 98.0)), 2)
    volt = round(float(np.clip(230.0 + 0.030*net + random.gauss(0, 0.20), 214.0, 246.0)), 2)
    h_l  = round(float(np.clip(0.0012 + abs(net)*0.00004 + random.gauss(0,0.00015), 0.0006, 0.0048)), 4)
    logs = []
    if weather == "Sudden Cloud Cover" and random.random() < 0.60:
        logs.append(("WARN", ts_str, f"Solar drop ({solar:.1f} kW). Dispatching 15 kW from BESS. Throttling non-critical loads."))
    if weather == "Special Function/Ceremony Unexpected Load Spike" and random.random() < 0.55:
        logs.append(("DISPATCH", ts_str, f"Ceremony surge ({load:.1f} kW). HNN |dH/dt|={h_l:.4f} pu. Max BESS discharge → 230 V stabilization."))
    if override and random.random() < 0.50:
        logs.append(("OVERRIDE", ts_str, "MANUAL OVERRIDE: Forced BESS discharge. Non-critical loads isolated (EV, HVAC). Critical circuits secured."))
    if not radio and random.random() < 0.40:
        logs.append(("WARN", ts_str, "Substation radio offline. Fallback to autonomous islanded HNN edge governor."))
    if random.random() < 0.65:
        r_txt = "connected" if radio else "offline (local HNN)"
        logs.append(("HNN", ts_str, f"Radio {r_txt}. HNN stable |dH/dt|={h_l:.4f} pu. Dispatch: {abs(net):.1f} kW via BESS."))
    return {"feeder_voltage": volt, "solar_gen": solar, "bess_soc": bess,
            "current_load": load, "h_loss": h_l, "new_logs": logs}


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PLOTLY HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _cl(height=340, **kw):
    """Shared chart layout."""
    legend_bg = f"rgba({17 if DARK else 248},{24 if DARK else 250},{39 if DARK else 252},0.90)"
    return dict(
        height=height,
        margin=dict(l=12, r=12, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(family="Inter, sans-serif", size=11, color=T["text_sec"]),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            font=dict(size=10, color=T["text_primary"]),
            bgcolor=legend_bg, bordercolor=T["border"], borderwidth=1,
        ),
        **kw,
    )


def _ax(title="", tc=None, **kw):
    """Shared axis style — callers can override any key via kw."""
    base = dict(
        title_text=title,
        title_font=dict(color=tc or T["text_muted"], size=10.5),
        tickfont=dict(color=T["text_muted"], size=10),
        showgrid=True,
        gridcolor=T["grid"],
        linecolor=T["axis"],
        tickcolor=T["axis"],
    )
    base.update(kw)   # kw wins — allows showgrid=False etc.
    return base


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-badge">AMG-2056</div>
        <div class="sb-ttl">Micro-Grid Governor</div>
        <div class="sb-sub">Decentralized Feeder Control</div>
    </div>""", unsafe_allow_html=True)

    # ── Theme Toggle
    st.markdown('<div class="sb-sec">🎨 Display Theme</div>', unsafe_allow_html=True)
    theme_new = st.toggle(
        "Dark Mode",
        value=st.session_state["dark_mode"],
        key="theme_toggle",
        help="Toggle between dark SCADA HMI and light engineering display.",
    )
    if theme_new != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = theme_new
        st.rerun()

    # ── Telemetry
    st.markdown('<div class="sb-sec">1 · Substation Telemetry</div>', unsafe_allow_html=True)
    radio_val = st.toggle(
        "Radio Transmission (SCADA Link)",
        value=st.session_state["radio_connected"],
        key="radio_toggle",
    )
    st.session_state["radio_connected"] = radio_val
    if radio_val:
        st.markdown('<div class="pill pill-g" style="width:100%;justify-content:center;margin-bottom:8px;"><span class="dot dg"></span> Radio: SCADA Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill pill-a" style="width:100%;justify-content:center;margin-bottom:8px;"><span class="dot da"></span> Offline: Local HNN Mode</div>', unsafe_allow_html=True)

    # ── Scenario
    st.markdown('<div class="sb-sec">2 · Climate & Event Simulator</div>', unsafe_allow_html=True)
    SCENARIOS = ["Normal", "Sudden Cloud Cover", "Special Function/Ceremony Unexpected Load Spike"]
    wx = st.radio("Simulation Scenario", options=SCENARIOS,
                  index=SCENARIOS.index(st.session_state["weather"]), key="wx_radio")
    st.session_state["weather"] = wx

    # ── Override
    st.markdown('<div class="sb-sec">3 · Emergency Demand Switch</div>', unsafe_allow_html=True)
    ov_val = st.toggle("Manual Override (Forced Shed)", value=st.session_state["manual_override"], key="override_toggle")
    st.session_state["manual_override"] = ov_val
    if ov_val:
        st.markdown('<div class="pill pill-r" style="width:100%;justify-content:center;margin-bottom:8px;">⚠ Forced BESS Discharge + Non-Critical Shed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill pill-c" style="width:100%;justify-content:center;margin-bottom:8px;">✓ Autonomous HNN Optimization</div>', unsafe_allow_html=True)

    # ── Buttons
    st.markdown('<div class="sb-sec">4 · Simulation Controls</div>', unsafe_allow_html=True)
    sb1, sb2 = st.columns(2)
    with sb1:
        if st.button("▶ Step", width='stretch'):
            td = sim_tick(st.session_state["weather"], st.session_state["manual_override"], st.session_state["radio_connected"])
            st.session_state.update({k: td[k] for k in ["feeder_voltage","solar_gen","bess_soc","current_load","h_loss"]})
            st.session_state["log_entries"].extend(td["new_logs"])
            st.session_state["tick"] += 1
            st.session_state["pinn_epoch"] += 2
            st.rerun()
    with sb2:
        if st.button("↺ Reset", width='stretch'):
            for k, v in _DEFAULTS.items():
                if k != "dark_mode":
                    st.session_state[k] = v
            st.rerun()

    auto_on = st.toggle("Live Auto-Run", value=st.session_state["auto_refresh"], key="auto_toggle")
    st.session_state["auto_refresh"] = auto_on
    rate = st.slider("Refresh Rate (s)", 2, 10, st.session_state["refresh_interval"])
    st.session_state["refresh_interval"] = rate

    with st.expander("⚙ Feeder Technical Specs", expanded=False):
        st.markdown("""
        - **Sub-Station**: 11 kV / 415 V 3-Phase (100 kVA Dyn11)
        - **Feeder**: R/X = 1.85  (95mm² Al XLPE cable)
        - **BESS**: 50 kWh / 25 kW LiFePO₄
        - **Inverter**: Bidirectional 4-quadrant converter
        - **HNN**: dH/dt = ∂H/∂q·q̇ + ∂H/∂p·ṗ = 0
        - **Code**: EN 50160  (230 V ±5%)
        """)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  AUTO-REFRESH TICK
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state["auto_refresh"]:
    td = sim_tick(st.session_state["weather"], st.session_state["manual_override"], st.session_state["radio_connected"])
    st.session_state.update({k: td[k] for k in ["feeder_voltage","solar_gen","bess_soc","current_load","h_loss"]})
    st.session_state["log_entries"].extend(td["new_logs"])
    st.session_state["log_entries"] = st.session_state["log_entries"][-60:]
    st.session_state["tick"] += 1
    st.session_state["pinn_epoch"] += 1

V   = st.session_state["feeder_voltage"]
S   = st.session_state["solar_gen"]
B   = st.session_state["bess_soc"]
L   = st.session_state["current_load"]
H_L = st.session_state["h_loss"]
NET = round(S - L, 2)
V_OK = 218.5 <= V <= 241.5


# ──────────────────────────────────────────────────────────────────────────────
# 9.  PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────
_rp = f'<span class="pill {"pill-c" if st.session_state["radio_connected"] else "pill-a"}"><span class="dot {"dc" if st.session_state["radio_connected"] else "da"}"></span> {"SCADA Online" if st.session_state["radio_connected"] else "Offline HNN"}</span>'
_vp = f'<span class="pill {"pill-g" if V_OK else "pill-r"}"><span class="dot {"dg" if V_OK else "dr"}"></span> {"Voltage Nominal" if V_OK else "Voltage Breach"}</span>'
_wp = {"Normal": '<span class="pill pill-g">🌞 Nominal</span>',
       "Sudden Cloud Cover": '<span class="pill pill-a">☁ Solar Intermittency</span>',
       "Special Function/Ceremony Unexpected Load Spike": '<span class="pill pill-r">🎉 Ceremony Surge</span>'}[st.session_state["weather"]]

st.markdown(f"""
<div class="amg-hdr">
  <div style="display:flex;align-items:center;gap:14px;">
    <div class="amg-icon">⚡</div>
    <div>
      <div class="amg-title">Autonomous Micro-Grid Governor &nbsp;<span style="color:#10B981;font-size:1.0rem;">(AMG-2056)</span></div>
      <div class="amg-sub">Neighbourhood Sub-Distributor &bull; PINN &amp; Hamiltonian Neural Network Intermittency Control</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    {_rp} {_wp} {_vp}
    <span class="pill pill-c">⏱ Tick #{st.session_state["tick"]}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 10.  KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(label, icon, value_html, delta_html, note, accent):
    return f"""
<div class="kpi">
  <div class="kpi-bar" style="background:{accent};"></div>
  <div class="kpi-lbl">{icon}&nbsp;&nbsp;{label}</div>
  <div class="kpi-val">{value_html}</div>
  <div class="kpi-foot">{delta_html}<span style="color:var(--tm);">{note}</span></div>
</div>"""

v_diff   = V - 230.0
v_dcl    = "kdg" if abs(v_diff) <= 5 else ("kdr" if abs(v_diff) > 11.5 else "kda")
b_color  = C["emerald"] if B >= 40 else (C["amber"] if B >= 20 else C["rose"])
b_state  = "Exporting" if NET > 2 else ("Absorbing" if NET < -2 else "Floating")
s_label  = "High Gen" if S > 18 else ("Moderate" if S > 5 else "Low / Night")

k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    st.markdown(kpi_card(
        "Feeder Voltage", "⚡",
        f'{V:.1f} <span>V</span>',
        f'<span class="kdelta {v_dcl}">{v_diff:+.1f} V from 230 V</span>',
        "EN 50160 ±5%",
        C["cyan"] if V_OK else C["rose"],
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(
        "Solar Generation", "☀",
        f'{S:.2f} <span>kW</span>',
        f'<span class="kdelta {"kdg" if S > 10 else "kdc"}">{s_label}</span>',
        "Rooftop Array · 35 kWp",
        C["amber"],
    ), unsafe_allow_html=True)
with k3:
    h_ok = H_L <= 0.0025
    st.markdown(kpi_card(
        "Hamiltonian State", "🧠",
        f'{H_L:.4f} <span>pu</span>',
        f'<span class="kdelta {"kdg" if h_ok else "kda"}">|dH/dt| {"≤" if h_ok else ">"} 0.0025</span>',
        "Symplectic Invariant",
        C["indigo"],
    ), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card(
        "BESS State of Charge", "🔋",
        f'<span style="color:{b_color};">{B:.1f}</span> <span>%</span>',
        f'<span class="kdelta kdc">{b_state} &nbsp;·&nbsp; {B*0.5:.1f} kWh</span>',
        "Reserve Limit: 20%",
        b_color,
    ), unsafe_allow_html=True)

st.markdown('<div class="gap14"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 11.  CHARTS ROW
# ──────────────────────────────────────────────────────────────────────────────
cc1, cc2 = st.columns([1.2, 0.8], gap="small")

with cc1:
    with st.container(border=True):
        st.markdown("""
        <div class="sec-hdr">
          <div class="sec-ttl">📈 Predicted vs. Actual Load &amp; Solar Generation</div>
          <div class="sec-sub">24-Hour Horizon &bull; HNN Forecasting</div>
        </div>""", unsafe_allow_html=True)

        df24  = gen_24h_data(st.session_state["weather"])
        fig_t = make_subplots(specs=[[{"secondary_y": True}]])

        for yhi, ylo, fc, sy in [
            (df24["HNN Predicted Load"],  df24["Actual Load"],  "rgba(6,182,212,0.07)",  False),
            (df24["HNN Predicted Solar"], df24["Actual Solar"], "rgba(245,158,11,0.07)", True),
        ]:
            fig_t.add_trace(go.Scatter(
                x=list(df24["Time"]) + list(df24["Time"])[::-1],
                y=list(yhi) + list(ylo)[::-1],
                fill="toself", fillcolor=fc,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), secondary_y=sy)

        for nm, y, col, w, dash, sy in [
            ("Actual Load (kW)",  df24["Actual Load"],         "#0D6EFD", 2.4, "solid", False),
            ("HNN Pred Load",     df24["HNN Predicted Load"],  "#74B0FF", 1.6, "dash",  False),
            ("Actual Solar (kW)", df24["Actual Solar"],        "#FD7E14", 2.4, "solid", True),
            ("HNN Pred Solar",    df24["HNN Predicted Solar"], "#FFB366", 1.6, "dot",   True),
        ]:
            fig_t.add_trace(go.Scatter(
                x=df24["Time"], y=y, name=nm,
                line=dict(color=col, width=w, dash=dash),
            ), secondary_y=sy)

        fig_t.update_layout(**_cl(340))
        fig_t.update_xaxes(**_ax(tickformat="%H:%M"))
        fig_t.update_yaxes(**_ax("Demand (kW)", tc="#0D6EFD"), secondary_y=False)
        fig_t.update_yaxes(**_ax("Solar (kW)", tc="#FD7E14", showgrid=False), secondary_y=True)
        st.plotly_chart(fig_t, width='stretch', config={"displayModeBar": False})

with cc2:
    with st.container(border=True):
        st.markdown(f"""
        <div class="sec-hdr">
          <div class="sec-ttl">⚡ Hamiltonian Convergence &amp; Loss</div>
          <div class="sec-sub">Epoch #{st.session_state["pinn_epoch"]} &bull; Symplectic Conservation</div>
        </div>""", unsafe_allow_html=True)

        df_l  = gen_loss_data(st.session_state["pinn_epoch"])
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=df_l["Epoch"], y=df_l["Total"], name="Total Symplectic Loss",
            line=dict(color="#28A745", width=2.4), fill="tozeroy", fillcolor="rgba(40,167,69,0.09)"))
        fig_l.add_trace(go.Scatter(x=df_l["Epoch"], y=df_l["H_error"], name="Conservation Error |dH/dt|",
            line=dict(color="#0D6EFD", width=1.8)))
        fig_l.add_trace(go.Scatter(x=df_l["Epoch"], y=df_l["MSE"], name="Data MSE Loss",
            line=dict(color="#FD7E14", width=1.8, dash="dash")))
        fig_l.add_hline(y=0.015, line=dict(color="#DC3545", width=1.2, dash="dot"),
            annotation_text="ε=0.015 pu", annotation_position="top right",
            annotation_font_color="#DC3545", annotation_font_size=9)
        fig_l.update_layout(**_cl(340, yaxis_type="log"))
        fig_l.update_xaxes(**_ax("Training Epoch"))
        fig_l.update_yaxes(**_ax("Loss (Log Scale)"))
        st.plotly_chart(fig_l, width='stretch', config={"displayModeBar": False})

st.markdown('<div class="gap14"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 12.  BOTTOM ROW:  Event Log  +  BESS Reserve Monitor (Plotly)
# ──────────────────────────────────────────────────────────────────────────────
bc1, bc2 = st.columns([1.2, 0.8], gap="small")

# ── 12a. Live Event Stream
with bc1:
    with st.container(border=True):
        st.markdown("""
        <div class="sec-hdr">
          <div class="sec-ttl">📋 Micro-Controller Autonomous Event Stream</div>
          <div class="sec-sub">AMG-2056 Edge Governor SCADA Feed</div>
        </div>""", unsafe_allow_html=True)

        _TAG = {
            "HNN":      ("t-hnn", "HNN-OK"),
            "DISPATCH": ("t-dsp", "DISPATCH"),
            "WARN":     ("t-wrn", "WARN"),
            "OVERRIDE": ("t-ovr", "OVERRIDE"),
        }
        lines = []
        for lvl, ts_l, msg in reversed(st.session_state["log_entries"]):
            cls, lbl = _TAG.get(lvl, ("t-hnn", lvl))
            lines.append(
                f'<div class="tl">'
                f'<span class="tts">[{ts_l}]</span>'
                f'<span class="ttag {cls}">{lbl}</span>'
                f'<span class="tmsg">{msg}</span>'
                f'</div>'
            )
        if not lines:
            lines.append('<div class="tl"><span class="tmsg">Initializing AMG-2056 telemetry…</span></div>')

        st.markdown(f'<div class="term">{"".join(lines)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="gap10"></div>', unsafe_allow_html=True)


# ── 12b. BESS Reserve Monitor  (Plotly gauge + tiles)
with bc2:
    with st.container(border=True):
        st.markdown("""
        <div class="sec-hdr">
          <div class="sec-ttl">🔋 Community BESS Reserve Monitor</div>
          <div class="sec-sub">50 kWh LiFePO₄ &bull; Real-Time Battery State</div>
        </div>""", unsafe_allow_html=True)

        b_hex   = C["emerald"] if B >= 40 else (C["amber"] if B >= 20 else C["rose"])
        b_act   = "Discharging" if NET < 0 else ("Charging" if NET > 0 else "Floating")

        # ── PLOTLY GAUGE (the actual visual monitor)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=B,
            number=dict(
                suffix="%",
                font=dict(family="JetBrains Mono", size=36, color=b_hex),
            ),
            delta=dict(
                reference=70.0,
                increasing=dict(color=C["emerald"]),
                decreasing=dict(color=C["rose"]),
                font=dict(size=13),
            ),
            title=dict(
                text=(
                    f"STATE OF CHARGE<br>"
                    f"<span style='font-size:11px;color:{T['text_muted']}'>"
                    f"{B*0.5:.1f} kWh / 50 kWh &nbsp;—&nbsp; {b_act}"
                    f"</span>"
                ),
                font=dict(size=12, color=T["text_sec"]),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100], tickwidth=1, tickcolor=T["text_muted"],
                    tickvals=[0, 20, 50, 70, 95, 100],
                    ticktext=["0", "Cutoff\n20%", "Buffer\n50%", "Target\n70%", "95%", "100"],
                    tickfont=dict(size=9, color=T["text_muted"]),
                ),
                bar=dict(color=b_hex, thickness=0.26),
                bgcolor=T["bg_inner"],
                borderwidth=1,
                bordercolor=T["border"],
                steps=[
                    dict(range=[0,  20], color="rgba(244,63,94,0.18)"),
                    dict(range=[20, 50], color="rgba(245,158,11,0.12)"),
                    dict(range=[50, 70], color="rgba(6,182,212,0.12)"),
                    dict(range=[70, 95], color="rgba(16,185,129,0.14)"),
                ],
                threshold=dict(
                    line=dict(color=C["rose"], width=3),
                    thickness=0.78, value=20,
                ),
            ),
        ))
        fig_gauge.update_layout(
            height=230,
            margin=dict(l=28, r=28, t=16, b=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=T["text_sec"]),
        )
        st.plotly_chart(fig_gauge, width='stretch', config={"displayModeBar": False})

        # ── Diagnostic tiles — 2×2 clean HTML grid (pixel-perfect alignment)
        st.markdown(f"""
        <div class="bess-grid">
          <div class="bess-tile">
            <span class="bess-tile-lbl">DC Bus Voltage</span>
            <span class="bess-tile-val" style="color:{C["cyan"]}">402.5 <span style="font-size:0.78rem;font-weight:600;">V</span></span>
          </div>
          <div class="bess-tile">
            <span class="bess-tile-lbl">Internal Temp</span>
            <span class="bess-tile-val" style="color:{C["amber"]}">26.8 <span style="font-size:0.78rem;font-weight:600;">°C</span></span>
          </div>
          <div class="bess-tile">
            <span class="bess-tile-lbl">Inverter SOH</span>
            <span class="bess-tile-val" style="color:{C["emerald"]}">98.4 <span style="font-size:0.78rem;font-weight:600;">%</span></span>
          </div>
          <div class="bess-tile">
            <span class="bess-tile-lbl">C-Rate</span>
            <span class="bess-tile-val" style="color:{C["indigo"]}">0.38 <span style="font-size:0.78rem;font-weight:600;">C</span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 13.  FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 8px 0;
    font-size:0.71rem;color:{T["text_muted"]};
    border-top:1px solid {T["border"]};margin-top:18px;">
  <strong style="color:{T["text_sec"]};">Autonomous Micro-Grid Governor (AMG-2056)</strong>
  &bull; Decentralised Energy Storage &amp; Voltage Regulation
  &bull; Python &middot; Streamlit &middot; Plotly
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 14.  LIVE REFRESH TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state["auto_refresh"]:
    time.sleep(st.session_state["refresh_interval"])
    st.rerun()
