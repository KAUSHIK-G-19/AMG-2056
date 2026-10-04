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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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
    # ── Panel visibility flags (toggled via sidebar)
    "show_3d":          False,
    "show_connections": False,
    "show_maintenance": False,
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
        # Substation Control Room — low-glare industrial dark slate (ISA-101 / IEEE 1613)
        "bg_page":          "#090D12",
        "bg_card":          "#111722",
        "bg_card_grad":     "linear-gradient(180deg, #151D28 0%, #0E141E 100%)",
        "bg_header_grad":   "linear-gradient(180deg, #16202E 0%, #0E1521 100%)",
        "bg_inner":         "#0B1018",
        "bg_terminal":      "#06090D",
        "bg_tile_grad":     "linear-gradient(180deg, #131A25 0%, #0D131B 100%)",
        "bg_table_hdr":     "linear-gradient(180deg, #151F2C 0%, #0E141E 100%)",
        "btn_bg_grad":      "linear-gradient(180deg, #182330 0%, #101722 100%)",
        "btn_hover_grad":   "linear-gradient(180deg, #1E2D3E 0%, #131D2A 100%)",
        "border":           "#1B2636",
        "border_hi":        "#2E3F57",
        # Luminescent high-contrast readouts — standard for 24/7 control rooms
        "text_primary":     "#E1EBF5",
        "text_sec":         "#91A6BF",
        "text_muted":       "#586F87",
        "plot_bg":          "#0B1017",
        "grid":             "#16202D",
        "axis":             "#2A3A50",
        "shadow":           "0 4px 18px rgba(0,0,0,0.65)",
        "shadow_card":      "0 2px 10px rgba(0,0,0,0.40)",
        "sidebar_bg":       "#0C1118",
        "term_text":        "#E1EBF5",
    }
    C = {
        "emerald": "#00E676",   # Substation Healthy / Breaker Safe Green
        "cyan":    "#00B4D8",   # SCADA Telemetry & Feeder Busbar Cyan
        "amber":   "#FFB300",   # 11 kV Busbar & Warning Annunciator Amber
        "rose":    "#EF4444",   # Breaker Trip & Voltage Breach Alarm Red
        "indigo":  "#818CF8",   # Hamiltonian Neural Core / Physics Link
        "violet":  "#A78BFA",
        # Chart trace colors (optimized for dark low-glare)
        "chart_load_act":   "#00B4D8",
        "chart_load_pred":  "#38BDF8",
        "chart_solar_act":  "#FFB300",
        "chart_solar_pred": "#FCD34D",
        "chart_loss_tot":   "#00E676",
        "chart_loss_h":     "#00B4D8",
        "chart_loss_mse":   "#FFB300",
        "chart_fill_load":  "rgba(0, 180, 216, 0.08)",
        "chart_fill_solar": "rgba(255, 179, 0, 0.08)",
        "chart_fill_loss":  "rgba(0, 230, 118, 0.10)",
        # Annunciator pill & delta badge styling
        "pill_g_bg": "linear-gradient(180deg, rgba(0, 230, 118, 0.22) 0%, rgba(0, 230, 118, 0.08) 100%)",
        "pill_g_tx": "#00E676",
        "pill_g_bd": "rgba(0, 230, 118, 0.60)",
        "pill_c_bg": "linear-gradient(180deg, rgba(0, 180, 216, 0.22) 0%, rgba(0, 180, 216, 0.08) 100%)",
        "pill_c_tx": "#00B4D8",
        "pill_c_bd": "rgba(0, 180, 216, 0.60)",
        "pill_a_bg": "linear-gradient(180deg, rgba(255, 179, 0, 0.22) 0%, rgba(255, 179, 0, 0.08) 100%)",
        "pill_a_tx": "#FFB300",
        "pill_a_bd": "rgba(255, 179, 0, 0.60)",
        "pill_r_bg": "linear-gradient(180deg, rgba(239, 68, 68, 0.24) 0%, rgba(239, 68, 68, 0.08) 100%)",
        "pill_r_tx": "#EF4444",
        "pill_r_bd": "rgba(239, 68, 68, 0.60)",
    }
else:
    T = {
        # Substation Daytime Engineering Report format (clean, harmonious, high contrast)
        "bg_page":          "#F1F5F9",
        "bg_card":          "#FFFFFF",
        "bg_card_grad":     "linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)",
        "bg_header_grad":   "linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%)",
        "bg_inner":         "#F8FAFC",
        "bg_terminal":      "#0F172A",    # Real SCADA SOE consoles maintain dark high-contrast box
        "bg_tile_grad":     "linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)",
        "bg_table_hdr":     "linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%)",
        "btn_bg_grad":      "linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)",
        "btn_hover_grad":   "linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%)",
        "border":           "#CBD5E1",
        "border_hi":        "#94A3B8",
        "text_primary":     "#0F172A",
        "text_sec":         "#334155",
        "text_muted":       "#64748B",
        "plot_bg":          "#FFFFFF",
        "grid":             "#E2E8F0",
        "axis":             "#94A3B8",
        "shadow":           "0 2px 12px rgba(15,23,42,0.08)",
        "shadow_card":      "0 1px 4px rgba(15,23,42,0.06)",
        "sidebar_bg":       "#F8FAFC",
        "term_text":        "#E2E8F0",
    }
    C = {
        # High-contrast engineering grade colors for crisp white background
        "emerald": "#15803D",   # Forest Green (4.8:1 contrast on white)
        "cyan":    "#0284C7",   # Deep Technical Blue (4.5:1 contrast on white)
        "amber":   "#B45309",   # Rich Bronze/Amber (4.7:1 contrast on white)
        "rose":    "#DC2626",   # Crisp Signal Red (5.1:1 contrast on white)
        "indigo":  "#4338CA",   # Deep Cobalt Indigo (7.2:1 contrast on white)
        "violet":  "#7C3AED",
        # Chart trace colors (optimized for white canvas - NEVER washed out or invisible!)
        "chart_load_act":   "#0284C7",
        "chart_load_pred":  "#0369A1",
        "chart_solar_act":  "#D97706",
        "chart_solar_pred": "#B45309",
        "chart_loss_tot":   "#15803D",
        "chart_loss_h":     "#0284C7",
        "chart_loss_mse":   "#B45309",
        "chart_fill_load":  "rgba(2, 132, 199, 0.08)",
        "chart_fill_solar": "rgba(217, 119, 6, 0.08)",
        "chart_fill_loss":  "rgba(21, 128, 61, 0.09)",
        # Annunciator pill & delta badge styling (high contrast)
        "pill_g_bg": "linear-gradient(180deg, rgba(22, 163, 74, 0.14) 0%, rgba(22, 163, 74, 0.06) 100%)",
        "pill_g_tx": "#15803D",
        "pill_g_bd": "rgba(22, 163, 74, 0.50)",
        "pill_c_bg": "linear-gradient(180deg, rgba(2, 132, 199, 0.14) 0%, rgba(2, 132, 199, 0.06) 100%)",
        "pill_c_tx": "#0369A1",
        "pill_c_bd": "rgba(2, 132, 199, 0.50)",
        "pill_a_bg": "linear-gradient(180deg, rgba(217, 119, 6, 0.14) 0%, rgba(217, 119, 6, 0.06) 100%)",
        "pill_a_tx": "#B45309",
        "pill_a_bd": "rgba(217, 119, 6, 0.50)",
        "pill_r_bg": "linear-gradient(180deg, rgba(220, 38, 38, 0.14) 0%, rgba(220, 38, 38, 0.06) 100%)",
        "pill_r_tx": "#B91C1C",
        "pill_r_bd": "rgba(220, 38, 38, 0.50)",
    }

# 4.  GLOBAL CSS  (injected once per render)
# ──────────────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Barlow+Semi+Condensed:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&family=Share+Tech+Mono&display=swap');

:root {{
    color-scheme: {"dark" if DARK else "light"};
    --bg-page:        {T["bg_page"]};
    --bg-card:        {T["bg_card"]};
    --bg-inner:       {T["bg_inner"]};
    --border:         {T["border"]};
    --border-hi:      {T["border_hi"]};
    --tp:             {T["text_primary"]};
    --ts:             {T["text_sec"]};
    --tm:             {T["text_muted"]};
    --green:  {C["emerald"]}; --blue:  {C["cyan"]};
    --orange: {C["amber"]};   --red:   {C["rose"]}; --purple: {C["indigo"]};
}}

html, body, .stApp {{
    font-family: 'Barlow', 'Segoe UI', Arial, sans-serif !important;
    background-color: var(--bg-page) !important;
    color: var(--tp) !important;
    font-size: 14px;
    line-height: 1.5 !important;
    letter-spacing: 0;
}}

.block-container {{
    padding: 1rem 2rem 2.5rem 2rem !important;
    max-width: 1440px !important;
}}

/* ── Remove annoying double-click arrow hints, modebars & sidebar arrows ── */
[data-testid="stSidebarCollapseButton"] {{
    display: none !important;
}}
.modebar, .plotly-notifier, .hoverlayer .tip {{
    display: none !important;
}}
[data-testid="stPlotlyChart"] .modebar-container {{
    display: none !important;
}}
.js-plotly-plot .plotly .cursor-crosshair,
.js-plotly-plot .plotly .cursor-pointer {{
    cursor: default !important;
}}

/* ── Substation Instrument Enclosures ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {T["bg_card_grad"]} !important;
    border: 1px solid {T["border"]} !important;
    border-top: 1px solid {T["border_hi"]} !important;
    border-radius: 10px !important;
    box-shadow: {T["shadow"]} !important;
}}

/* ── Substation Page Header Banner ── */
.amg-hdr {{
    background: {T["bg_header_grad"]};
    border: 1px solid {T["border"]};
    border-top: 1px solid {T["border_hi"]};
    border-radius: 10px;
    padding: 16px 22px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 14px; margin-bottom: 18px;
    box-shadow: {T["shadow"]}, inset 0 1px 0 rgba(255,255,255,0.07);
}}
.amg-icon {{
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #B45309 0%, #F59E0B 50%, #D97706 100%);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-size: 24px; color: #fff;
    box-shadow: 0 0 18px rgba(245,158,11,0.40), inset 0 1px 0 rgba(255,255,255,0.30);
    border: 1px solid #FCD34D;
    flex-shrink: 0;
}}
.amg-title {{
    font-family: 'Barlow', sans-serif;
    font-size: 1.25rem; font-weight: 700; color: var(--tp);
    line-height: 1.35; letter-spacing: 0.02em;
    margin-bottom: 3px;
}}
.amg-sub {{
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 0.78rem; color: var(--tm); font-weight: 500;
    letter-spacing: 0.03em; text-transform: uppercase;
    line-height: 1.4;
}}

/* ── Substation Annunciator Status Tiles (Backlit LED Bevels) ── */
.pill {{
    display: inline-flex; align-items: center; gap: 7px;
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 0.74rem; font-weight: 700; padding: 5px 12px;
    border-radius: 4px; white-space: normal; line-height: 1.35;
    letter-spacing: 0.05em; text-transform: uppercase;
    word-break: break-word;
}}
.pill-g {{
    background: {C["pill_g_bg"]};
    color: {C["pill_g_tx"]};
    border: 1px solid {C["pill_g_bd"]};
    box-shadow: inset 0 0 8px rgba(0, 200, 83, 0.12);
}}
.pill-c {{
    background: {C["pill_c_bg"]};
    color: {C["pill_c_tx"]};
    border: 1px solid {C["pill_c_bd"]};
    box-shadow: inset 0 0 8px rgba(2, 132, 199, 0.12);
}}
.pill-a {{
    background: {C["pill_a_bg"]};
    color: {C["pill_a_tx"]};
    border: 1px solid {C["pill_a_bd"]};
    box-shadow: inset 0 0 8px rgba(245, 158, 11, 0.12);
}}
.pill-r {{
    background: {C["pill_r_bg"]};
    color: {C["pill_r_tx"]};
    border: 1px solid {C["pill_r_bd"]};
    box-shadow: inset 0 0 8px rgba(239, 68, 68, 0.14);
}}
.dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.dg {{ background: {C["emerald"]}; box-shadow: 0 0 8px {C["emerald"]}; }}
.dc {{ background: {C["cyan"]};    box-shadow: 0 0 8px {C["cyan"]}; }}
.da {{ background: {C["amber"]};   box-shadow: 0 0 8px {C["amber"]}; }}
.dr {{ background: {C["rose"]};    box-shadow: 0 0 8px {C["rose"]}; }}

/* ── Substation KPI Rack Instruments (No overlapping, flexible height) ── */
.kpi {{
    min-height: 156px;
    height: auto;
    background: {T["bg_card_grad"]};
    border: 1px solid {T["border"]};
    border-top: 1px solid {T["border_hi"]};
    border-radius: 8px; padding: 14px 16px;
    display: flex; flex-direction: column; justify-content: space-between;
    gap: 8px;
    box-shadow: {T["shadow_card"]}, inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color .18s ease, box-shadow .18s ease;
    box-sizing: border-box; overflow: visible; position: relative;
}}
.kpi:hover {{
    border-color: {T["border_hi"]};
    box-shadow: {T["shadow"]};
}}
.kpi-bar {{ position: absolute; top: 0; left: 0; width: 4px; height: 100%; border-radius: 8px 0 0 8px; }}
.kpi-lbl {{
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 0.76rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--ts);
    line-height: 1.35; margin: 0;
}}
.kpi-val {{
    font-family: 'JetBrains Mono', 'Share Tech Mono', monospace;
    font-size: 1.70rem; font-weight: 700; line-height: 1.25; color: var(--tp);
    letter-spacing: -0.01em; margin: 3px 0;
    word-break: normal;
    white-space: nowrap;
}}
.kpi-val span {{ font-size: 0.90rem; color: var(--tm); font-weight: 500; margin-left: 3px; }}
.kpi-foot {{
    display: flex; flex-direction: column; align-items: flex-start;
    gap: 6px;
    font-size: 0.72rem; color: var(--ts);
    font-family: 'Barlow', sans-serif;
    line-height: 1.35;
    margin-top: auto;
}}
.kdelta {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 7px; border-radius: 4px; font-weight: 700;
    font-size: 0.69rem; font-family: 'JetBrains Mono', monospace;
    line-height: 1.25;
    white-space: normal;
    word-break: break-word;
}}
.kdg {{ background: {C["pill_g_bg"]}; color: {C["pill_g_tx"]}; border: 1px solid {C["pill_g_bd"]}; }}
.kda {{ background: {C["pill_a_bg"]}; color: {C["pill_a_tx"]}; border: 1px solid {C["pill_a_bd"]}; }}
.kdc {{ background: {C["pill_c_bg"]}; color: {C["pill_c_tx"]}; border: 1px solid {C["pill_c_bd"]}; }}
.kdr {{ background: {C["pill_r_bg"]}; color: {C["pill_r_tx"]}; border: 1px solid {C["pill_r_bd"]}; }}

/* ── Panel Section Header (Machined Bevel) ── */
.sec-hdr {{
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 8px;
    padding: 13px 16px 10px 16px;
    background: {T["bg_card_grad"]};
    border-bottom: 1px solid {T["border"]}; flex-shrink: 0;
}}
.sec-ttl {{
    font-family: 'Barlow', sans-serif;
    font-size: 0.88rem; font-weight: 700; color: var(--tp);
    display: flex; align-items: center; flex-wrap: wrap; gap: 7px;
    letter-spacing: 0.02em; line-height: 1.3;
    min-width: 0; overflow-wrap: anywhere;
}}
.sec-sub {{
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 0.72rem; color: var(--tm); font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; line-height: 1.3;
}}

/* ── Substation Sequence of Events (SOE) Terminal Log ── */
.term {{
    height: 320px;
    overflow-y: scroll;
    font-family: 'JetBrains Mono', 'Share Tech Mono', monospace;
    background: {T["bg_terminal"]}; border-radius: 6px;
    margin: 12px 14px 12px 14px; padding: 10px 12px;
    font-size: 0.71rem; line-height: 1.85;
    border: 1px solid {T["border_hi"]};
    border-left: 4px solid #00B4D8;
    scrollbar-width: thin;
    scrollbar-color: #0284C7 transparent;
}}
.term::-webkit-scrollbar {{ width: 6px; }}
.term::-webkit-scrollbar-track {{ background: transparent; border-radius: 4px; }}
.term::-webkit-scrollbar-thumb {{ background: #0284C7; border-radius: 4px; }}
.term::-webkit-scrollbar-thumb:hover {{ background: #38BDF8; }}
.tl {{
    display: flex; align-items: flex-start; gap: 8px;
    padding: 3px 6px; border-radius: 4px; margin-bottom: 4px;
}}
.tl:hover {{ background: rgba(255,255,255, 0.06); }}
.tts  {{ color: #94A3B8 !important; white-space: nowrap; flex-shrink: 0; font-size: 0.65rem; min-width: 68px; font-weight: 500; }}
.ttag {{
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 700; font-size: 0.64rem; padding: 2px 7px;
    border-radius: 3px; white-space: nowrap; flex-shrink: 0;
    min-width: 62px; text-align: center; letter-spacing: 0.05em;
}}
.t-hnn  {{ background: linear-gradient(180deg, rgba(0,180,216,.25) 0%, rgba(0,180,216,.10) 100%); color: #38BDF8; border: 1px solid rgba(0,180,216,.6); }}
.t-dsp  {{ background: linear-gradient(180deg, rgba(0,200,83,.25) 0%, rgba(0,200,83,.10) 100%); color: #00E676; border: 1px solid rgba(0,200,83,.6); }}
.t-wrn  {{ background: linear-gradient(180deg, rgba(245,158,11,.25) 0%, rgba(245,158,11,.10) 100%); color: #FBBF24; border: 1px solid rgba(245,158,11,.6); }}
.t-ovr  {{ background: linear-gradient(180deg, rgba(239,68,68,.25) 0%, rgba(239,68,68,.10) 100%); color: #F87171; border: 1px solid rgba(239,68,68,.6); }}
.tmsg {{ color: {T["term_text"]} !important; word-break: break-word; flex: 1; }}

/* ── BESS Diagnostic Grid (Recessed Substation Bay) ── */
.bess-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 0 14px 14px 14px;
}}
.bess-tile {{
    background: {T["bg_tile_grad"]};
    border: 1px solid {T["border"]};
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}}
.bess-tile-lbl {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tm);
    font-family: 'Barlow Semi Condensed', sans-serif;
    line-height: 1.3;
}}
.bess-tile-val {{
    font-family: 'JetBrains Mono', 'Share Tech Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--tp);
    line-height: 1.2;
}}

/* ── Substation Mimic Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {T["sidebar_bg"]} !important;
    border-right: 1px solid {T["border"]} !important;
}}
.sb-brand {{
    text-align: center; padding: 4px 0 16px 0;
    border-bottom: 1px solid {T["border"]}; margin-bottom: 16px;
}}
.sb-badge {{
    display: inline-block;
    background: linear-gradient(135deg, #B45309 0%, #F59E0B 100%);
    border: 1px solid #FCD34D;
    color: #FFFFFF; font-weight: 800; font-size: 0.78rem;
    padding: 4px 14px; border-radius: 4px; letter-spacing: 0.08em;
    margin-bottom: 6px; font-family: 'Barlow Semi Condensed', sans-serif;
    box-shadow: 0 0 12px rgba(245,158,11,0.30);
}}
.sb-ttl {{
    font-size: 0.98rem; font-weight: 700; color: var(--tp);
    font-family: 'Barlow', sans-serif; letter-spacing: 0.03em;
    line-height: 1.3;
}}
.sb-sub {{
    font-size: 0.69rem; color: var(--tm); text-transform: uppercase;
    letter-spacing: 0.09em; font-weight: 600;
    font-family: 'Barlow Semi Condensed', sans-serif;
}}
.sb-sec {{
    font-size: 0.72rem; font-weight: 800; color: #0284C7;
    text-transform: uppercase; letter-spacing: 0.10em;
    margin: 16px 0 7px 0; font-family: 'Barlow Semi Condensed', sans-serif;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border-hi); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #00B4D8; }}

.gap14 {{ height: 14px; }} .gap10 {{ height: 10px; }}

/* ── 3D Prototype Section ── */
.proto-wrap {{
    padding: 16px 16px 0 16px;
    display: flex; flex-direction: column; gap: 12px;
}}
.proto-upload-zone {{
    border: 2px dashed {T["border_hi"]};
    border-radius: 10px;
    padding: 28px 20px;
    text-align: center;
    background: {T["bg_inner"]};
    color: var(--tm);
    font-size: 0.82rem;
    transition: border-color .2s ease;
    font-family: 'Barlow', sans-serif;
}}
.proto-upload-zone:hover {{ border-color: #00B4D8; }}
.proto-upload-icon {{ font-size: 2.4rem; margin-bottom: 8px; }}
.proto-upload-ttl {{ font-size: 0.92rem; font-weight: 700; color: var(--tp); margin-bottom: 4px; font-family: 'Barlow', sans-serif; }}
.proto-upload-sub {{ font-size: 0.74rem; color: var(--tm); }}

/* ── Substation Hardware Connection List (Terminal Style) ── */
.conn-table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 0 14px 0;
}}
.conn-table th {{
    background: {T["bg_table_hdr"]};
    color: {T["text_sec"]};
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 8px 12px;
    border-bottom: 2px solid {T["border_hi"]};
    text-align: left;
    font-family: 'Barlow Semi Condensed', sans-serif;
}}
.conn-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {T["border"]};
    color: {T["text_primary"]};
    font-weight: 500;
    vertical-align: middle;
    overflow-wrap: anywhere;
}}
.conn-table th {{ overflow-wrap: anywhere; }}
.conn-table th:nth-child(2), .conn-table td:nth-child(2) {{
    width: 28px;
    text-align: center;
}}
.conn-table tr:hover td {{ background: {T["bg_inner"]}; }}
.pin-badge {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 0.69rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    white-space: normal;
    overflow-wrap: anywhere;
}}
.pin-5v   {{ background: linear-gradient(180deg, rgba(239,68,68,.20) 0%, rgba(239,68,68,.08) 100%);  color:#DC2626; border:1px solid rgba(239,68,68,.45); }}
.pin-3v3  {{ background: linear-gradient(180deg, rgba(245,158,11,.20) 0%, rgba(245,158,11,.08) 100%); color:#D97706; border:1px solid rgba(245,158,11,.45); }}
.pin-gnd  {{ background: linear-gradient(180deg, rgba(148,163,184,.20) 0%, rgba(148,163,184,.08) 100%);  color:#64748B; border:1px solid rgba(148,163,184,.45); }}
.pin-gpio {{ background: linear-gradient(180deg, rgba(2,132,199,.20) 0%, rgba(2,132,199,.08) 100%); color:#0284C7; border:1px solid rgba(2,132,199,.45); }}
.pin-spi  {{ background: linear-gradient(180deg, rgba(139,92,246,.20) 0%, rgba(139,92,246,.08) 100%); color:#7C3AED; border:1px solid rgba(139,92,246,.45); }}
.sensor-hdr {{
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px 6px 14px;
    font-size: 0.84rem; font-weight: 700; color: var(--tp);
    font-family: 'Barlow', sans-serif;
    border-bottom: 1px solid {T["border"]};
    margin-bottom: 2px;
}}
.sensor-icon {{
    width: 28px; height: 28px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}}
.conn-note {{
    font-size: 0.74rem; color: {T["text_primary"]};
    font-weight: 500;
    font-family: 'Barlow', sans-serif;
    background: {T["bg_inner"]};
    border: 1px solid {T["border"]};
    border-left: 3px solid #D97706;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 6px 14px 12px 14px;
    line-height: 1.45;
}}

/* ── Substation Maintenance & Safety Rules ── */
.maint-rule {{
    display: flex; gap: 12px; align-items: flex-start;
    padding: 11px 14px;
    border-bottom: 1px solid {T["border"]};
}}
.maint-rule:last-child {{ border-bottom: none; }}
.maint-rule-num {{
    min-width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #0284C7, #0369A1);
    border: 1px solid #38BDF8;
    color: #fff; font-size: 0.74rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
    box-shadow: 0 0 10px rgba(2,132,199,0.35);
}}
.maint-rule-ttl {{
    font-size: 0.84rem; font-weight: 700; color: {T["text_primary"]}; margin-bottom: 3px;
    font-family: 'Barlow', sans-serif; line-height: 1.35;
}}
.maint-rule-body {{
    font-size: 0.75rem; color: {T["text_sec"]}; line-height: 1.55;
    font-weight: 500;
    font-family: 'Barlow', sans-serif;
}}

.prob-card {{
    border: 1px solid {T["border"]};
    border-radius: 8px;
    overflow: hidden;
    margin: 8px 14px;
    background: {T["bg_card_grad"]};
}}
.prob-hdr {{
    display: flex; align-items: center; gap: 8px;
    padding: 9px 12px;
    background: {T["bg_table_hdr"]};
    border-bottom: 1px solid {T["border"]};
}}
.prob-hdr-icon {{ font-size: 1.1rem; flex-shrink: 0; }}
.prob-ttl {{
    font-size: 0.84rem; font-weight: 700; color: #DC2626;
    font-family: 'Barlow', sans-serif; line-height: 1.35;
}}
.prob-cause {{ font-size: 0.74rem; color: {T["text_sec"]}; padding: 8px 12px 0 12px; font-family: 'Barlow', sans-serif; line-height: 1.45; font-weight: 500; }}
.prob-cause strong {{ color: {T["text_primary"]}; font-weight: 700; }}
.prob-sol {{
    font-size: 0.75rem; color: {T["text_primary"]}; line-height: 1.55;
    font-weight: 500;
    padding: 6px 12px 10px 12px;
    border-top: 1px dashed {T["border"]};
    margin-top: 6px;
    font-family: 'Barlow', sans-serif;
}}
.prob-sol-lbl {{
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    background: {C["pill_g_bg"]};
    color: {C["pill_g_tx"]};
    border: 1px solid {C["pill_g_bd"]};
    font-size: 0.65rem; font-weight: 700;
    margin-bottom: 4px; letter-spacing: 0.08em;
    font-family: 'Barlow Semi Condensed', sans-serif;
}}

/* ═══════════════════════════════════════════════════════════════════════
   PER-THEME STREAMLIT WIDGET OVERRIDES
═══════════════════════════════════════════════════════════════════════ */

/* ── Streamlit app canvas & page background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main,
.main,
[data-testid="stHeader"] {{
    background-color: {T["bg_page"]} !important;
}}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stSidebar"] > div:first-child {{
    background-color: {T["sidebar_bg"]} !important;
    border-right: 1px solid {T["border"]} !important;
}}

/* ── Every native text node in the page ── */
.stApp p, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp li, .stApp small, .stApp strong,
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
.stRadio label p,
.stCheckbox label p,
[data-testid="stSlider"] p,
[data-testid="stSlider"] p,
.stSlider p,
[data-testid="stExpander"] p,
.streamlit-expanderHeader,
.streamlit-expanderContent p,
[data-testid="stSelectbox"] p,
[data-baseweb="select"] p {{
    color: {T["text_primary"]} !important;
    font-family: 'Barlow', sans-serif;
}}

/* ── Muted / helper text ── */
.stApp small,
[data-testid="stCaptionContainer"] p {{
    color: {T["text_muted"]} !important;
    font-family: 'Barlow', sans-serif;
}}

/* ── Secondary text (sub-labels) ── */
[data-testid="stWidgetLabel"] small {{
    color: {T["text_sec"]} !important;
    font-family: 'Barlow', sans-serif;
}}

/* ── Sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {T["text_primary"]} !important;
    font-family: 'Barlow', sans-serif;
}}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: {T["text_muted"]} !important;
    font-family: 'Barlow', sans-serif;
}}

/* ── Input & select backgrounds ── */
[data-baseweb="input"], [data-baseweb="select"],
[data-baseweb="textarea"] {{
    background-color: {T["bg_inner"]} !important;
    color: {T["text_primary"]} !important;
    border-color: {T["border"]} !important;
}}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
    background: {T["bg_inner"]} !important;
    border-color: {T["border"]} !important;
    color: {T["text_sec"]} !important;
}}
[data-testid="stFileUploader"] button {{
    background: {T["btn_bg_grad"]} !important;
    border: 1px solid {T["border_hi"]} !important;
    color: {T["text_primary"]} !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}
[data-testid="stFileUploader"] button:hover {{
    background: {T["btn_hover_grad"]} !important;
    border-color: {C["cyan"]} !important;
}}
[data-testid="stFileUploader"] button [data-testid="stIconMaterial"] {{
    font-family: "Material Symbols Rounded" !important;
    font-size: 1.1rem !important;
    font-weight: normal !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

.gap14 {{ height: 14px; }} .gap10 {{ height: 10px; }}

/* ── Substation Pushbutton Overrides ── */
.stButton > button,
.stButton > button[data-testid^="stBaseButton"] {{
    background: {T["btn_bg_grad"]} !important;
    color: {T["text_primary"]} !important;
    border: 1px solid {T["border"]} !important;
    border-top: 1px solid {T["border_hi"]} !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.48rem 0.95rem !important;
    line-height: 1.35 !important;
    white-space: normal !important;
    word-break: break-word !important;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: {T["shadow_card"]} !important;
}}
.stButton > button:hover,
.stButton > button[data-testid^="stBaseButton"]:hover {{
    background: {T["btn_hover_grad"]} !important;
    border-color: #00B4D8 !important;
    color: #0284C7 !important;
    box-shadow: 0 0 12px rgba(0, 180, 216, 0.35) !important;
}}
.stButton > button:active,
.stButton > button[data-testid^="stBaseButton"]:active {{
    background: #0284C7 !important;
    color: #FFFFFF !important;
}}
.stButton > button p,
.stButton > button span,
.stButton > button[data-testid^="stBaseButton"] p,
.stButton > button[data-testid^="stBaseButton"] span {{
    color: {T["text_primary"]} !important;
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    letter-spacing: 0.05em !important;
}}

/* Sidebar specific button styling */
[data-testid="stSidebar"] .stButton > button {{
    background: {T["btn_bg_grad"]} !important;
    border: 1px solid {T["border"]} !important;
    border-top: 1px solid {T["border_hi"]} !important;
    color: {T["text_primary"]} !important;
    margin-bottom: 6px !important;
    text-align: left !important;
    justify-content: flex-start !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {T["btn_hover_grad"]} !important;
    border-color: #00B4D8 !important;
    color: #0284C7 !important;
}}

/* Primary (Active) Button Styling — Annunciator Blue Glow */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(180deg, #0284C7 0%, #0369A1 100%) !important;
    border: 1px solid #38BDF8 !important;
    border-top: 1px solid #7DD3FC !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 14px rgba(2, 132, 199, 0.45), inset 0 1px 0 rgba(255,255,255,0.30) !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(180deg, #0369A1 0%, #075985 100%) !important;
    border-color: #BAE6FD !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 18px rgba(2, 132, 199, 0.60) !important;
}}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
button[data-testid="stBaseButton-primary"] p,
button[data-testid="stBaseButton-primary"] span,
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}

/* Panel Close Button (substation alarm red accent) */
.panel-close-btn-wrap .stButton > button {{
    background: linear-gradient(180deg, rgba(220, 38, 38, 0.14) 0%, rgba(220, 38, 38, 0.06) 100%) !important;
    border: 1px solid rgba(220, 38, 38, 0.50) !important;
    color: #DC2626 !important;
    justify-content: center !important;
    text-align: center !important;
    min-height: 38px !important;
}}
.panel-close-btn-wrap .stButton > button:hover {{
    background: linear-gradient(180deg, rgba(220, 38, 38, 0.28) 0%, rgba(220, 38, 38, 0.12) 100%) !important;
    border-color: #EF4444 !important;
    color: #EF4444 !important;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.35) !important;
}}
.panel-close-btn-wrap .stButton > button p,
.panel-close-btn-wrap .stButton > button span {{
    color: #DC2626 !important;
    font-weight: 800 !important;
}}

/* ── Streamlit Toggle Switch Styling (high-contrast text in white theme) ── */
[data-testid="stToggle"] label p,
[data-testid="stToggle"] label span,
[data-testid="stToggle"] span {{
    color: {T["text_primary"]} !important;
    font-family: 'Barlow', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    line-height: 1.4 !important;
}}
[data-testid="stToggle"] {{
    padding: 3px 0 !important;
}}

.amg-hdr > *, .sec-hdr > *, .kpi, .kpi > *, .maint-rule > div {{
    min-width: 0;
}}
.amg-title, .amg-sub, .sec-sub, .kpi-lbl, .kpi-foot,
.maint-rule-ttl, .maint-rule-body, .prob-ttl, .prob-cause, .prob-sol {{
    overflow-wrap: anywhere;
}}

@media (max-width: 768px) {{
    .block-container {{ padding: 0.75rem 0.75rem 2rem !important; }}
    .amg-hdr {{ align-items: flex-start; }}
    .kpi {{ padding: 12px; }}
    .kpi-val {{ font-size: 1.35rem; }}
    .conn-table {{ font-size: 0.68rem; }}
    .conn-table th, .conn-table td {{ padding: 6px 5px; }}
}}

</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

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
    legend_bg = "rgba(11, 16, 24, 0.92)" if DARK else "rgba(255, 255, 255, 0.95)"
    return dict(
        height=height,
        margin=dict(l=12, r=12, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(family="Barlow, sans-serif", size=11, color=T["text_sec"]),
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
        title_font=dict(family="Barlow Semi Condensed, sans-serif", color=tc or T["text_muted"], size=11),
        tickfont=dict(family="JetBrains Mono, monospace", color=T["text_muted"], size=10),
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
    st.toggle(
        "Dark Mode",
        key="dark_mode",
        help="Toggle between dark SCADA HMI and light engineering display.",
    )

    # ── Telemetry
    st.markdown('<div class="sb-sec">1 · Substation Telemetry</div>', unsafe_allow_html=True)
    radio_val = st.toggle("Radio Transmission (SCADA Link)", key="radio_connected")
    if radio_val:
        st.markdown('<div class="pill pill-g" style="width:100%;justify-content:center;margin-bottom:8px;"><span class="dot dg"></span> Radio: SCADA Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill pill-a" style="width:100%;justify-content:center;margin-bottom:8px;"><span class="dot da"></span> Offline: Local HNN Mode</div>', unsafe_allow_html=True)

    # ── Scenario
    st.markdown('<div class="sb-sec">2 · Climate & Event Simulator</div>', unsafe_allow_html=True)
    SCENARIOS = ["Normal", "Sudden Cloud Cover", "Special Function/Ceremony Unexpected Load Spike"]
    st.radio("Simulation Scenario", options=SCENARIOS, key="weather")

    # ── Override
    st.markdown('<div class="sb-sec">3 · Emergency Demand Switch</div>', unsafe_allow_html=True)
    ov_val = st.toggle("Manual Override (Forced Shed)", key="manual_override")
    if ov_val:
        st.markdown('<div class="pill pill-r" style="width:100%;justify-content:center;margin-bottom:8px;">⚠ Forced BESS Discharge + Non-Critical Shed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill pill-c" style="width:100%;justify-content:center;margin-bottom:8px;">✓ Autonomous HNN Optimization</div>', unsafe_allow_html=True)

    # ── Buttons
    st.markdown('<div class="sb-sec">4 · Simulation Controls</div>', unsafe_allow_html=True)
    sb1, sb2 = st.columns(2)
    with sb1:
        if st.button("▶ Step", use_container_width=True):
            td = sim_tick(st.session_state["weather"], st.session_state["manual_override"], st.session_state["radio_connected"])
            st.session_state.update({k: td[k] for k in ["feeder_voltage","solar_gen","bess_soc","current_load","h_loss"]})
            st.session_state["log_entries"].extend(td["new_logs"])
            st.session_state["tick"] += 1
            st.session_state["pinn_epoch"] += 2
            st.rerun()
    with sb2:
        if st.button("↺ Reset", use_container_width=True):
            for k, v in _DEFAULTS.items():
                if k != "dark_mode":
                    st.session_state[k] = v
            st.rerun()

    st.toggle("Live Auto-Run", key="auto_refresh")
    st.slider("Refresh Rate (s)", 2, 10, key="refresh_interval")

    with st.expander("⚙ Feeder Technical Specs", expanded=False):
        st.markdown("""
        - **Sub-Station**: 11 kV / 415 V 3-Phase (100 kVA Dyn11)
        - **Feeder**: R/X = 1.85  (95mm² Al XLPE cable)
        - **BESS**: 50 kWh / 25 kW LiFePO₄
        - **Inverter**: Bidirectional 4-quadrant converter
        - **HNN**: dH/dt = ∂H/∂q·q̇ + ∂H/∂p·ṗ = 0
        - **Code**: EN 50160  (230 V ±5%)
        """)

    # ── Panel Navigation
    st.markdown('<div class="sb-sec">5 · Dedicated Panel Navigator</div>', unsafe_allow_html=True)
    st.caption("Select / de-select options to display or dismiss panels:")

    st.toggle("🧪 3D Hardware Prototype", key="show_3d")
    st.toggle("🔌 Hardware Connections", key="show_connections")
    st.toggle("🛠️ Maintenance & Fixes", key="show_maintenance")


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
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <div class="amg-icon">⚡</div>
    <div>
      <div class="amg-title">Autonomous Micro-Grid Governor &nbsp;<span style="color:{C["emerald"]};font-size:1.02rem;font-weight:800;">(AMG-2056)</span></div>
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
            (df24["HNN Predicted Load"],  df24["Actual Load"],  C["chart_fill_load"],  False),
            (df24["HNN Predicted Solar"], df24["Actual Solar"], C["chart_fill_solar"], True),
        ]:
            fig_t.add_trace(go.Scatter(
                x=list(df24["Time"]) + list(df24["Time"])[::-1],
                y=list(yhi) + list(ylo)[::-1],
                fill="toself", fillcolor=fc,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), secondary_y=sy)

        for nm, y, col, w, dash, sy in [
            ("Actual Load (kW)",  df24["Actual Load"],         C["chart_load_act"],  2.4, "solid", False),
            ("HNN Pred Load",     df24["HNN Predicted Load"],  C["chart_load_pred"], 1.7, "dash",  False),
            ("Actual Solar (kW)", df24["Actual Solar"],        C["chart_solar_act"], 2.4, "solid", True),
            ("HNN Pred Solar",    df24["HNN Predicted Solar"], C["chart_solar_pred"], 1.8, "dot",   True),
        ]:
            fig_t.add_trace(go.Scatter(
                x=df24["Time"], y=y, name=nm,
                line=dict(color=col, width=w, dash=dash),
            ), secondary_y=sy)

        fig_t.update_layout(**_cl(340))
        fig_t.update_xaxes(**_ax(tickformat="%H:%M"))
        fig_t.update_yaxes(**_ax("Demand (kW)", tc=C["cyan"]), secondary_y=False)
        fig_t.update_yaxes(**_ax("Solar (kW)", tc=C["amber"], showgrid=False), secondary_y=True)
        st.plotly_chart(fig_t, width='stretch', config={"displayModeBar": False, "doubleClick": False, "showTips": False, "scrollZoom": False})

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
            line=dict(color=C["chart_loss_tot"], width=2.4), fill="tozeroy", fillcolor=C["chart_fill_loss"]))
        fig_l.add_trace(go.Scatter(x=df_l["Epoch"], y=df_l["H_error"], name="Conservation Error |dH/dt|",
            line=dict(color=C["chart_loss_h"], width=1.8)))
        fig_l.add_trace(go.Scatter(x=df_l["Epoch"], y=df_l["MSE"], name="Data MSE Loss",
            line=dict(color=C["chart_loss_mse"], width=1.8, dash="dash")))
        fig_l.add_hline(y=0.015, line=dict(color=C["rose"], width=1.2, dash="dot"),
            annotation_text="ε=0.015 pu", annotation_position="top right",
            annotation_font_color=C["rose"], annotation_font_size=9)
        fig_l.update_layout(**_cl(340, yaxis_type="log"))
        fig_l.update_xaxes(**_ax("Training Epoch"))
        fig_l.update_yaxes(**_ax("Loss (Log Scale)"))
        st.plotly_chart(fig_l, width='stretch', config={"displayModeBar": False, "doubleClick": False, "showTips": False, "scrollZoom": False})

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
            font=dict(family="Barlow, sans-serif", color=T["text_sec"]),
        )
        st.plotly_chart(fig_gauge, width='stretch', config={"displayModeBar": False, "doubleClick": False, "showTips": False, "scrollZoom": False})

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
# 13.  3-D PROTOTYPE VIEWER  (toggled via sidebar)
# ──────────────────────────────────────────────────────────────────────────────
def _close_panel(panel_key):
    st.session_state[panel_key] = False

if st.session_state.get("show_3d", False):
    st.markdown('<div class="gap14"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        _hdr_c, _cls_c = st.columns([6, 1])
        with _hdr_c:
            st.markdown("""
            <div class="sec-hdr" style="border-bottom:none;padding-bottom:4px;">
              <div class="sec-ttl">🧪 AMG-2056 Hardware Prototype — 3D View</div>
              <div class="sec-sub">Interactive Component Layout · ESP32 + Sensor Assembly</div>
            </div>""", unsafe_allow_html=True)
        with _cls_c:
            st.markdown('<div class="panel-close-btn-wrap">', unsafe_allow_html=True)
            st.button(
                "✕ Close",
                key="close_3d",
                help="Hide 3D Viewer panel",
                on_click=_close_panel,
                args=("show_3d",),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        proto_c1, proto_c2 = st.columns([1.6, 1.0], gap="small")

        with proto_c1:
            # PCB base mesh
            _xx = np.linspace(-3, 3, 20)
            _yy = np.linspace(-3, 3, 20)
            _XX, _YY = np.meshgrid(_xx, _yy)
            _ZZ = np.zeros_like(_XX)
            pcb_color = "#162438" if DARK else "#1A472A"

            components = [
                # (x,   y,    z,   label,         color,      size)
                ( 0.0,  0.0,  0.1, "ESP32",        "#0D6EFD",  18),
                (-1.8,  1.2,  0.2, "ZMPT101B",     "#FD7E14",  12),
                ( 1.8,  1.2,  0.2, "ACS712",       "#28A745",  12),
                (-1.8, -1.2,  0.2, "Relay IN1",    "#DC3545",  11),
                ( 1.8, -1.2,  0.2, "Relay IN2",    "#DC3545",  11),
                ( 0.0,  2.0,  0.3, "NRF24L01",     "#6F42C1",  13),
                (-0.7, -2.2,  0.1, "5V Supply",    "#FD7E14",  10),
                ( 0.7, -2.2,  0.1, "3.3V Reg",     "#FD7E14",  10),
            ]
            xs, ys, zs, labs, cols, szs = zip(*components)

            fig3d = go.Figure()
            # PCB board surface
            fig3d.add_trace(go.Surface(
                x=_XX, y=_YY, z=_ZZ,
                colorscale=[[0, pcb_color], [1, pcb_color]],
                showscale=False, opacity=0.85, hoverinfo="skip", name="PCB Board",
            ))
            # Wire traces from ESP32 to each component
            esp_x, esp_y, esp_z = 0.0, 0.0, 0.1
            for cx, cy, cz, lbl, col, _ in components[1:]:
                fig3d.add_trace(go.Scatter3d(
                    x=[esp_x, cx], y=[esp_y, cy], z=[esp_z + 0.05, cz + 0.05],
                    mode="lines",
                    line=dict(color=col, width=2),
                    showlegend=False, hoverinfo="skip",
                ))
            # Component markers
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=[v + 0.15 for v in zs],
                mode="markers+text",
                marker=dict(size=szs, color=cols, opacity=0.92,
                            line=dict(color="rgba(255,255,255,0.3)", width=1)),
                text=labs,
                textposition="top center",
                textfont=dict(size=10, color=T["text_primary"]),
                name="Components",
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))
            fig3d.update_layout(
                height=480,
                margin=dict(l=0, r=0, t=4, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    bgcolor=T["bg_inner"],
                    xaxis=dict(showgrid=False, showticklabels=False, title="",
                               backgroundcolor=T["bg_inner"], zeroline=False,
                               range=[-3.2, 3.2]),
                    yaxis=dict(showgrid=False, showticklabels=False, title="",
                               backgroundcolor=T["bg_inner"], zeroline=False,
                               range=[-3.2, 3.2]),
                    zaxis=dict(showgrid=False, showticklabels=False, title="",
                               backgroundcolor=T["bg_inner"], zeroline=False,
                               range=[0, 1.5]),
                    camera=dict(eye=dict(x=0.75, y=-0.75, z=0.65),
                                projection=dict(type="orthographic")),
                    aspectmode="manual",
                    aspectratio=dict(x=1.6, y=1.6, z=0.65),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig3d, width='stretch', config={"displayModeBar": False, "doubleClick": False, "showTips": False, "scrollZoom": True})

        with proto_c2:
            st.markdown(f"""
            <div class="proto-wrap">
              <div class="proto-upload-zone">
                <div class="proto-upload-icon">🏗️</div>
                <div class="proto-upload-ttl">Upload Custom 3D Model</div>
                <div class="proto-upload-sub">Drag &amp; drop your .STL / .OBJ / .STEP file here<br>
                to replace the default schematic view.</div>
              </div>
            </div>""", unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload 3D Model File",
                type=["stl", "obj", "step", "stp"],
                help="Upload an STL, OBJ, or STEP file of your prototype.",
                label_visibility="collapsed",
            )
            if uploaded_file:
                st.success(f"✅ **{uploaded_file.name}** uploaded! "
                           "3D rendering for STL/OBJ can be added via `trimesh` or `open3d`.")

            # Component colour legend
            legend_items = [
                ("#0D6EFD", "ESP32 Micro-Controller"),
                ("#FD7E14", "ZMPT101B Voltage Sensor"),
                ("#28A745", "ACS712 Current Sensor"),
                ("#DC3545", "2-Ch Relay Module"),
                ("#6F42C1", "NRF24L01 RF Transceiver"),
            ]
            legend_html = "".join(
                f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;'
                f'border-bottom:1px solid var(--border);font-size:0.75rem;color:var(--ts);">'
                f'<span style="width:12px;height:12px;border-radius:3px;background:{col};'
                f'display:inline-block;flex-shrink:0;"></span>{lbl}</div>'
                for col, lbl in legend_items
            )
            st.markdown(f"""
            <div style="padding:14px 4px 6px 4px;">
              <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.10em;
                          text-transform:uppercase;color:var(--tm);margin-bottom:6px;">
                Component Legend
              </div>
              {legend_html}
            </div>
            <div style="font-size:0.70rem;color:var(--tm);padding:8px 4px 0 4px;line-height:1.6;">
              💡 <strong style="color:var(--ts);">Tip:</strong> Drag to rotate &bull;
              Scroll to zoom &bull; Right-click to pan. Lines = GPIO wire connections from ESP32.
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 14.  HARDWARE CONNECTION LIST  (toggled via sidebar)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("show_connections", False):
    st.markdown('<div class="gap14"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        _hdr_cc, _cls_cc = st.columns([6, 1])
        with _hdr_cc:
            st.markdown("""
            <div class="sec-hdr" style="border-bottom:none;padding-bottom:4px;">
              <div class="sec-ttl">🔌 Hardware Connection List — ESP32 Pin Mapping</div>
              <div class="sec-sub">Verified Wiring Reference &bull; AMG-2056 Prototype v1.0</div>
            </div>""", unsafe_allow_html=True)
        with _cls_cc:
            st.markdown('<div class="panel-close-btn-wrap">', unsafe_allow_html=True)
            st.button(
                "✕ Close",
                key="close_conn",
                help="Hide Hardware Connections panel",
                on_click=_close_panel,
                args=("show_connections",),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        conn_c1, conn_c2 = st.columns(2, gap="small")

        with conn_c1:
            st.markdown(f"""
            <div class="sensor-hdr">
              <div class="sensor-icon" style="background:rgba(253,126,20,.18);">⚡</div>
              <span>ZMPT101B &mdash; Voltage Sensor</span>
            </div>
            <table class="conn-table">
              <thead><tr>
                <th>Sensor Pin</th><th>→</th><th>ESP32 Pin</th><th>Note</th>
              </tr></thead>
              <tbody>
                <tr><td>VCC</td><td>→</td><td><span class="pin-badge pin-5v">5 V</span></td><td>Power supply</td></tr>
                <tr><td>GND</td><td>→</td><td><span class="pin-badge pin-gnd">GND</span></td><td>Common ground</td></tr>
                <tr><td>OUT (Signal)</td><td>→</td><td><span class="pin-badge pin-gpio">GPIO 34</span></td><td>ADC1_CH6 · Analog In</td></tr>
              </tbody>
            </table>

            <div class="sensor-hdr">
              <div class="sensor-icon" style="background:rgba(40,167,69,.18);">〰️</div>
              <span>ACS712 &mdash; Current Sensor</span>
            </div>
            <table class="conn-table">
              <thead><tr>
                <th>Sensor Pin</th><th>→</th><th>ESP32 Pin</th><th>Note</th>
              </tr></thead>
              <tbody>
                <tr><td>VCC</td><td>→</td><td><span class="pin-badge pin-5v">5 V</span></td><td>Power supply</td></tr>
                <tr><td>GND</td><td>→</td><td><span class="pin-badge pin-gnd">GND</span></td><td>Common ground</td></tr>
                <tr><td>OUT (Signal)</td><td>→</td><td><span class="pin-badge pin-gpio">GPIO 35</span></td><td>ADC1_CH7 · Analog In</td></tr>
              </tbody>
            </table>
            """, unsafe_allow_html=True)

        with conn_c2:
            st.markdown(f"""
            <div class="sensor-hdr">
              <div class="sensor-icon" style="background:rgba(220,53,69,.18);">🔁</div>
              <span>2-Channel Relay Module</span>
            </div>
            <table class="conn-table">
              <thead><tr>
                <th>Module Pin</th><th>→</th><th>ESP32 Pin</th><th>Note</th>
              </tr></thead>
              <tbody>
                <tr><td>VCC</td><td>→</td><td><span class="pin-badge pin-5v">5 V</span></td><td>Power supply</td></tr>
                <tr><td>GND</td><td>→</td><td><span class="pin-badge pin-gnd">GND</span></td><td>Common ground</td></tr>
                <tr><td>IN1</td><td>→</td><td><span class="pin-badge pin-gpio">GPIO 25</span></td><td>Non-Critical Load Shift</td></tr>
                <tr><td>IN2</td><td>→</td><td><span class="pin-badge pin-gpio">GPIO 26</span></td><td>Community BESS Control</td></tr>
              </tbody>
            </table>

            <div class="sensor-hdr">
              <div class="sensor-icon" style="background:rgba(111,66,193,.18);">📡</div>
              <span>NRF24L01 &mdash; RF Transceiver</span>
            </div>
            <div class="conn-note">⚠ Use <strong>3.3 V only</strong> — powering with 5 V will permanently damage the module.</div>
            <table class="conn-table">
              <thead><tr>
                <th>Module Pin</th><th>→</th><th>ESP32 Pin</th><th>Note</th>
              </tr></thead>
              <tbody>
                <tr><td>VCC</td><td>→</td><td><span class="pin-badge pin-3v3">3.3 V</span></td><td>⚠ NOT 5 V</td></tr>
                <tr><td>GND</td><td>→</td><td><span class="pin-badge pin-gnd">GND</span></td><td>Common ground</td></tr>
                <tr><td>CE</td><td>→</td><td><span class="pin-badge pin-gpio">GPIO 4</span></td><td>Chip Enable</td></tr>
                <tr><td>CSN</td><td>→</td><td><span class="pin-badge pin-gpio">GPIO 5</span></td><td>Chip Select</td></tr>
                <tr><td>SCK</td><td>→</td><td><span class="pin-badge pin-spi">GPIO 18</span></td><td>SPI Clock</td></tr>
                <tr><td>MOSI</td><td>→</td><td><span class="pin-badge pin-spi">GPIO 23</span></td><td>SPI Data Out</td></tr>
                <tr><td>MISO</td><td>→</td><td><span class="pin-badge pin-spi">GPIO 19</span></td><td>SPI Data In</td></tr>
              </tbody>
            </table>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 15.  MAINTENANCE & TROUBLESHOOTING  (toggled via sidebar)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("show_maintenance", False):
    st.markdown('<div class="gap14"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        _hdr_m, _cls_m = st.columns([6, 1])
        with _hdr_m:
            st.markdown("""
            <div class="sec-hdr" style="border-bottom:none;padding-bottom:4px;">
              <div class="sec-ttl">🛠️ Prototype Maintenance &amp; Troubleshooting Centre</div>
              <div class="sec-sub">Team Rules &bull; Calibration Protocols &bull; Known Issues &amp; Fixes</div>
            </div>""", unsafe_allow_html=True)
        with _cls_m:
            st.markdown('<div class="panel-close-btn-wrap">', unsafe_allow_html=True)
            st.button(
                "✕ Close",
                key="close_maint",
                help="Hide Maintenance panel",
                on_click=_close_panel,
                args=("show_maintenance",),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        maint_c1, maint_c2 = st.columns([1.0, 1.0], gap="small")

        with maint_c1:
            st.markdown("""
            <div style="padding:10px 14px 4px 14px;font-size:0.75rem;font-weight:700;
                        letter-spacing:0.08em;text-transform:uppercase;color:var(--tm);
                        border-bottom:1px solid var(--border);">
              📋 Maintenance Team Rules
            </div>
            """, unsafe_allow_html=True)

            rules = [
                ("Pre-Deployment Calibration Check",
                 "Always calibrate the voltage and current sensor offsets using baseline test runs "
                 "before mounting the hardware onto the feeder box."),
                ("Firmware Version Control",
                 "Maintain clean, documented code updates on GitHub. Any emergency hotfixes applied "
                 "on-site must be pushed to the repository immediately."),
                ("Power Isolation Protocol",
                 "Always disconnect main power supplies before inspecting or rewiring the 2-channel "
                 "relay module or sensor terminals to prevent short circuits."),
                ("Routine Communication Audit",
                 "Weekly test the local radio transceivers (NRF24L01/RF links) to ensure uninterrupted "
                 "weather and telemetry data exchange between the substation and the edge governor."),
                ("Battery Health Monitoring (BESS)",
                 "Periodically log Community BESS state-of-charge (SoC) parameters to ensure local "
                 "buffering reliability during sudden high-load scenarios."),
            ]
            rules_html = "".join(
                f'<div class="maint-rule">'
                f'<div class="maint-rule-num">{i+1}</div>'
                f'<div><div class="maint-rule-ttl">{ttl}</div>'
                f'<div class="maint-rule-body">{body}</div></div>'
                f'</div>'
                for i, (ttl, body) in enumerate(rules)
            )
            st.markdown(f'<div>{rules_html}</div>', unsafe_allow_html=True)

        with maint_c2:
            st.markdown("""
            <div style="padding:10px 14px 4px 14px;font-size:0.75rem;font-weight:700;
                        letter-spacing:0.08em;text-transform:uppercase;color:var(--tm);
                        border-bottom:1px solid var(--border);">
              ⚠️ Frequently Occurred Problems &amp; Solutions
            </div>
            """, unsafe_allow_html=True)

            problems = [
                (
                    "Sensor Reading Fluctuations / Noise Interference",
                    "Electromagnetic interference (EMI) near power lines affecting analog pins.",
                    "Implement software-level moving average digital filters in the ESP32 code and "
                    "use shielded wires for sensor analog output pins.",
                ),
                (
                    "Radio Signal Drop / Communication Failure",
                    "Physical obstructions or interference blocking the RF transceiver frequency.",
                    "Built-in fail-safe kicks in — if signal drops for >10 s, hardware switches to "
                    "local hardcoded rule-based safety limits to maintain uninterrupted baseline power.",
                ),
                (
                    "ESP32 Restarts Unexpectedly (Brownout Detector Triggered)",
                    "Power drop when both relays and sensors draw high current from the MCU board.",
                    "Power the 2-channel relay module and sensors via an independent, stable external "
                    "5 V supply instead of drawing heavy current directly from the ESP32 5 V pin.",
                ),
                (
                    "Relay Chattering / Rapid Switching",
                    "Voltage or power values hovering right around the threshold limit.",
                    "Add a time-delay hysteresis buffer in the decision logic so relays only trigger "
                    "after sustained deviations exceeding the threshold for at least 3–5 seconds.",
                ),
            ]
            for prob, cause, sol in problems:
                st.markdown(f"""
                <div class="prob-card">
                  <div class="prob-hdr">
                    <span class="prob-hdr-icon">🔴</span>
                    <span class="prob-ttl">{prob}</span>
                  </div>
                  <div class="prob-cause"><strong>Cause:</strong> {cause}</div>
                  <div class="prob-sol">
                    <span class="prob-sol-lbl">✓ SOLUTION</span><br>{sol}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# 16.  FOOTER
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
# 17.  LIVE REFRESH TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("auto_refresh", False):
    _interval = max(1, int(st.session_state.get("refresh_interval", 3)))
    st_autorefresh(interval=_interval * 1000, key="amg_auto_refresh")



