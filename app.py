"""
SIMULADOR BURSÁTIL — Maestría en Economía y Finanzas
=====================================================
Aplicación principal Streamlit.

Módulos integrados:
  M1 — Motor Matemático (CETES, UDIBONOS, VaR, Duración)
  M2 — Modelado de Instrumentos (Black-Scholes, FIBRAS, ETFs, Frontera Eficiente)
  M3 — Simulador de Crisis (4 escenarios históricos)
  M4 — IA Predictiva (Random Forest)
  M5 — Visualización Científica (Plotly)

Ejecutar:
    streamlit run app.py

Feria de Innovación — Tercer Cuatrimestre
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Asegurar que los módulos sean encontrados
sys.path.insert(0, os.path.dirname(__file__))

from modulo1.motor_matematico import (
    precio_cetes, rendimiento_real_cetes, tabla_cetes,
    precio_udibono, valor_udi, comparar_cetes_udibono,
    duracion_macaulay, duracion_modificada, convexidad,
    cambio_precio_bono, var_parametrico,
)
from modulo2.modelado_instrumentos import (
    black_scholes, valuar_convertible, sensibilidad_convertible_crisis,
    frontera_eficiente,
)
from modulo3.simulador_crisis import (
    aplicar_crisis, comparar_todas_las_crisis,
    evolucion_temporal_crisis, PORTAFOLIO_BASE, CRISIS_CONFIG,
)
from modulo4.ia_predictiva import (
    entrenar_modelo, predecir_activo_resiliente, importancia_variables_df,
)
from modulo5.visualizacion import (
    grafica_evolucion_portafolio, grafica_impacto_crisis,
    grafica_frontera_eficiente, grafica_ilusion_monetaria,
    grafica_anatomia_convertible, grafica_correlaciones,
    grafica_comparativa_crisis, grafica_gauge_riesgo,
)


# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Simulador Bursátil MEF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design System: Dark Mode OLED · IBM Plex Sans · Fintech Palette ──────────
st.markdown("""
<style>
/* ── Google Fonts: IBM Plex Sans + Material Symbols ────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ── Fix ícono sidebar (double_arrow_right) ─────────────────────────────────── */
[data-testid="collapsedControl"] span {
    font-family: 'Material Symbols Rounded' !important;
}

/* ── Variables de color — Dark pero con más vida ────────────────────────────── */
:root {
    --bg-base:      #0A0F1E;
    --bg-card:      #111827;
    --bg-card-2:    #1F2937;
    --bg-card-3:    #263348;
    --border:       rgba(255,255,255,0.10);
    --border-hover: rgba(34,197,94,0.5);
    --text-primary: #F1F5F9;
    --text-muted:   #94A3B8;
    --text-subtle:  #64748B;
    --green:        #22C55E;
    --green-light:  #4ADE80;
    --green-dim:    rgba(34,197,94,0.15);
    --green-glow:   rgba(34,197,94,0.30);
    --red:          #F87171;
    --red-dim:      rgba(248,113,113,0.15);
    --blue:         #60A5FA;
    --blue-dark:    #3B82F6;
    --blue-dim:     rgba(96,165,250,0.15);
    --teal:         #2DD4BF;
    --teal-dim:     rgba(45,212,191,0.15);
    --amber:        #FBBF24;
    --amber-dim:    rgba(251,191,36,0.15);
    --purple:       #C084FC;
    --purple-dim:   rgba(192,132,252,0.15);
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    16px;
    --shadow-card:  0 4px 24px rgba(0,0,0,0.35);
    --transition:   all 0.2s cubic-bezier(0.4,0,0.2,1);
}

/* ── Tipografía global ─────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp, .main, section, .block-container,
h1, h2, h3, h4, h5, h6, p, span, div, label, button {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif !important;
}
code, pre, .stCode { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Fondo base con gradiente sutil ─────────────────────────────────────────── */
.stApp {
    background: linear-gradient(160deg, #0A0F1E 0%, #0D1628 50%, #0A1628 100%) !important;
    background-attachment: fixed !important;
}
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ── Sidebar con acento azul ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0D1628 100%) !important;
    border-right: 1px solid rgba(96,165,250,0.15) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text-primary) !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: var(--text-muted) !important; font-size: 0.85rem; }

/* ── HEADER PRINCIPAL ──────────────────────────────────────────────────────── */
.mef-header {
    position: relative;
    background: linear-gradient(135deg, #111827 0%, #0D1B3E 40%, #0F2044 70%, #111827 100%);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(59,130,246,0.08), 0 4px 24px rgba(0,0,0,0.4);
}
.mef-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(96,165,250,0.10) 0%, rgba(34,197,94,0.05) 60%, transparent 80%);
    pointer-events: none;
}
.mef-header-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(96,165,250,0.15), rgba(34,197,94,0.15));
    border: 1px solid rgba(96,165,250,0.4);
    color: var(--blue);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.mef-header h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #F1F5F9 0%, #60A5FA 50%, #4ADE80 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem !important;
}
.mef-header .subtitle {
    color: #94A3B8;
    font-size: 1rem;
    margin: 0 0 0.5rem;
}
.mef-header .tags {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem;
    margin-top: 0.8rem;
}
.mef-header .tag {
    background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.2);
    color: #93C5FD;
    font-size: 0.73rem;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-weight: 500;
}

/* ── TABS ──────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(96,165,250,0.15) !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1rem !important;
    transition: var(--transition) !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(96,165,250,0.08) !important;
    color: #93C5FD !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1F2937 0%, #1a2e45 100%) !important;
    color: #60A5FA !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.3), 0 0 0 1px rgba(96,165,250,0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── METRIC CARDS ──────────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1a2538 100%) !important;
    border: 1px solid rgba(96,165,250,0.15) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(96,165,250,0.4) !important;
    box-shadow: 0 0 20px rgba(96,165,250,0.15) !important;
    transform: translateY(-1px);
}
div[data-testid="stMetricLabel"] { color: #93C5FD !important; font-size: 0.8rem !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }
div[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── BOTONES STREAMLIT ─────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-card-2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.2rem !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--bg-card) !important;
    border-color: var(--green) !important;
    box-shadow: 0 0 16px var(--green-glow) !important;
    color: var(--green) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0) scale(0.98) !important; }

/* ── INPUTS / SLIDERS ──────────────────────────────────────────────────────── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: var(--bg-card-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}
div[data-testid="stSelectbox"] > div {
    background: var(--bg-card-2) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--green) !important;
    border-color: var(--green) !important;
}

/* ── DATAFRAMES ────────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── KPI CARDS CUSTOM ──────────────────────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1rem;
    text-align: center;
    transition: var(--transition);
    cursor: default;
}
.kpi-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 0 20px var(--green-glow);
    transform: translateY(-2px);
}
.kpi-card .etiqueta {
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-subtle);
    margin-bottom: 0.4rem;
}
.kpi-card .valor {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.2rem 0;
    color: var(--text-primary);
}
.kpi-card .delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.2rem;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    display: inline-block;
}
.kpi-positivo .valor { color: var(--green); }
.kpi-positivo .delta { background: var(--green-dim); color: var(--green); }
.kpi-negativo .valor { color: var(--red); }
.kpi-negativo .delta { background: var(--red-dim); color: var(--red); }
.kpi-neutro   .valor { color: var(--blue); }
.kpi-neutro   .delta { background: var(--blue-dim); color: var(--blue); }

/* ── BOTONES DE CRISIS ─────────────────────────────────────────────────────── */
.crisis-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0 1.5rem;
}
.crisis-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 0.75rem;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
    user-select: none;
}
.crisis-btn:hover    { transform: translateY(-2px); box-shadow: var(--shadow-card); }
.crisis-btn:active   { transform: scale(0.97); }
.crisis-btn .btn-icon { font-size: 1.8rem; display: block; margin-bottom: 0.4rem; }
.crisis-btn .btn-title { font-size: 0.82rem; font-weight: 700; color: var(--text-primary); display: block; }
.crisis-btn .btn-sub  { font-size: 0.72rem; color: var(--text-muted); display: block; margin-top: 0.15rem; }

.crisis-btn.fuego  { border-color: rgba(239,68,68,0.35);  }
.crisis-btn.fuego:hover  { background: rgba(239,68,68,0.08); border-color: var(--red); box-shadow: 0 0 20px rgba(239,68,68,0.2); }
.crisis-btn.virus  { border-color: rgba(168,85,247,0.35); }
.crisis-btn.virus:hover  { background: rgba(168,85,247,0.08); border-color: var(--purple); box-shadow: 0 0 20px rgba(168,85,247,0.2); }
.crisis-btn.bomba  { border-color: rgba(245,158,11,0.35); }
.crisis-btn.bomba:hover  { background: rgba(245,158,11,0.08); border-color: var(--amber); box-shadow: 0 0 20px rgba(245,158,11,0.2); }
.crisis-btn.rayo   { border-color: rgba(59,130,246,0.35);  }
.crisis-btn.rayo:hover   { background: rgba(59,130,246,0.08); border-color: var(--blue); box-shadow: 0 0 20px rgba(59,130,246,0.2); }

.crisis-btn.activo { box-shadow: 0 0 0 2px currentColor; }
.crisis-fuego-activo  { background: rgba(239,68,68,0.1)  !important; border-color: var(--red)    !important; }
.crisis-virus-activo  { background: rgba(168,85,247,0.1) !important; border-color: var(--purple) !important; }
.crisis-bomba-activo  { background: rgba(245,158,11,0.1) !important; border-color: var(--amber)  !important; }
.crisis-rayo-activo   { background: rgba(59,130,246,0.1)  !important; border-color: var(--blue)   !important; }

/* ── CAJAS DE CONTENIDO ────────────────────────────────────────────────────── */
.concepto-box {
    background: linear-gradient(135deg, rgba(96,165,250,0.10) 0%, rgba(45,212,191,0.06) 100%);
    border: 1px solid rgba(96,165,250,0.28);
    border-left: 3px solid var(--blue);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #E2E8F0;
    line-height: 1.6;
}
.alerta-roja {
    background: linear-gradient(135deg, rgba(248,113,113,0.12) 0%, rgba(239,68,68,0.06) 100%);
    border: 1px solid rgba(248,113,113,0.30);
    border-left: 3px solid var(--red);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #FEE2E2;
    line-height: 1.6;
}
.alerta-verde {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(74,222,128,0.06) 100%);
    border: 1px solid rgba(34,197,94,0.30);
    border-left: 3px solid var(--green);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #DCFCE7;
    line-height: 1.6;
}
.alerta-amber {
    background: linear-gradient(135deg, rgba(251,191,36,0.12) 0%, rgba(245,158,11,0.06) 100%);
    border: 1px solid rgba(251,191,36,0.30);
    border-left: 3px solid var(--amber);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #FEF3C7;
    line-height: 1.6;
}

/* ── TARJETAS DE ACTIVO EN CRISIS ──────────────────────────────────────────── */
.activo-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.9rem 1rem;
    margin: 0.4rem 0;
    transition: var(--transition);
    cursor: default;
}
.activo-card:hover { border-color: rgba(255,255,255,0.15); transform: translateY(-1px); }
.activo-card .ac-nombre  { font-size: 0.82rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.2rem; }
.activo-card .ac-impacto { font-size: 1.1rem; font-weight: 700; }
.activo-card .ac-valor   { font-size: 0.78rem; color: var(--text-muted); }
.activo-card .ac-texto   { font-size: 0.74rem; color: var(--text-subtle); margin-top: 0.3rem; line-height: 1.4; }
.activo-positivo { border-left: 3px solid var(--green); }
.activo-positivo .ac-impacto { color: var(--green); }
.activo-negativo { border-left: 3px solid var(--red); }
.activo-negativo .ac-impacto { color: var(--red); }

/* ── BADGE DE ESTADO ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    text-transform: uppercase;
}
.badge-green  { background: var(--green-dim); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
.badge-red    { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(239,68,68,0.3); }
.badge-blue   { background: var(--blue-dim);  color: var(--blue);  border: 1px solid rgba(59,130,246,0.3); }
.badge-amber  { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }

/* ── SECCIÓN HEADERS ───────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-header h2, .section-header h3 {
    margin: 0 !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    flex-shrink: 0;
}

/* ── DIVIDER ───────────────────────────────────────────────────────────────── */
hr, .stDivider { border-color: var(--border) !important; }

/* ── PROGRESO ──────────────────────────────────────────────────────────────── */
div[data-testid="stProgress"] > div > div {
    background: var(--green) !important;
    border-radius: 999px !important;
}
div[data-testid="stProgress"] > div {
    background: var(--bg-card-2) !important;
    border-radius: 999px !important;
}

/* ── SPINNER ───────────────────────────────────────────────────────────────── */
.stSpinner { color: var(--green) !important; }

/* ── CAPTION / INFO ────────────────────────────────────────────────────────── */
.stCaption, caption { color: var(--text-subtle) !important; font-size: 0.78rem !important; }
.stInfo    { background: var(--blue-dim)  !important; border-color: rgba(59,130,246,0.3) !important; }
.stSuccess { background: var(--green-dim) !important; border-color: rgba(34,197,94,0.3)  !important; }
.stWarning { background: var(--amber-dim) !important; border-color: rgba(245,158,11,0.3) !important; }
.stError   { background: var(--red-dim)   !important; border-color: rgba(239,68,68,0.3)  !important; }

/* ── RADIO ─────────────────────────────────────────────────────────────────── */
div[data-testid="stRadio"] label { color: var(--text-muted) !important; font-size: 0.85rem !important; }

/* ── SCROLLBAR PERSONALIZADO ───────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-card-2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-subtle); }

/* ── FOOTER ────────────────────────────────────────────────────────────────── */
.mef-footer {
    text-align: center;
    color: var(--text-subtle);
    font-size: 0.78rem;
    padding: 1.5rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.mef-footer strong { color: var(--text-muted); }

/* ── RESPONSIVE MÓVIL ──────────────────────────────────────────────────────── */
@media (max-width: 768px) {

    /* Contenedor principal sin padding lateral excesivo */
    .block-container {
        padding: 0.75rem 0.5rem 1rem !important;
        max-width: 100% !important;
    }

    /* Header compacto */
    .mef-header {
        padding: 1.5rem 1rem 1.2rem !important;
        margin-bottom: 1rem !important;
    }
    .mef-header h1 {
        font-size: 1.5rem !important;
        letter-spacing: -0.01em !important;
    }
    .mef-header .subtitle {
        font-size: 0.82rem !important;
    }
    .mef-header .tags {
        gap: 0.3rem !important;
    }
    .mef-header .tag {
        font-size: 0.68rem !important;
        padding: 0.15rem 0.5rem !important;
    }

    /* Tabs: scrollables en horizontal */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        padding: 3px !important;
        gap: 1px !important;
        -webkit-overflow-scrolling: touch;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.75rem !important;
        padding: 0.4rem 0.65rem !important;
        white-space: nowrap !important;
    }

    /* KPI grid: 2 columnas en móvil */
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 0.5rem !important;
    }
    .kpi-card .valor {
        font-size: 1.4rem !important;
    }

    /* Crisis buttons: 2 columnas en móvil */
    .crisis-grid {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 0.5rem !important;
    }
    .crisis-btn {
        padding: 0.85rem 0.5rem !important;
    }
    .crisis-btn .btn-icon { font-size: 1.4rem !important; }
    .crisis-btn .btn-title { font-size: 0.75rem !important; }
    .crisis-btn .btn-sub  { font-size: 0.68rem !important; }

    /* Métricas nativas de Streamlit */
    div[data-testid="stMetric"] {
        padding: 0.75rem 0.85rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }

    /* Activo cards: texto más compacto */
    .activo-card .ac-nombre  { font-size: 0.78rem !important; }
    .activo-card .ac-impacto { font-size: 1rem !important; }
    .activo-card .ac-valor   { font-size: 0.74rem !important; }
    .activo-card .ac-texto   { font-size: 0.70rem !important; }

    /* Concepto/alerta boxes */
    .concepto-box, .alerta-roja, .alerta-verde, .alerta-amber {
        font-size: 0.82rem !important;
        padding: 0.7rem 0.85rem !important;
    }

    /* Botones más grandes para touch (44px mínimo) */
    .stButton > button {
        min-height: 44px !important;
        font-size: 0.85rem !important;
        padding: 0.7rem 1rem !important;
    }

    /* Inputs táctiles */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        min-height: 44px !important;
        font-size: 1rem !important;  /* evita zoom en iOS */
    }

    /* Sliders más accesibles */
    div[data-testid="stSlider"] {
        padding: 0.25rem 0 !important;
    }

    /* Sidebar: ocupa pantalla completa en móvil (comportamiento Streamlit default) */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        font-size: 0.9rem !important;
    }

    /* Footer */
    .mef-footer {
        font-size: 0.72rem !important;
        padding: 1rem !important;
    }
}

/* ── TABLET (768px–1024px) ─────────────────────────────────────────────────── */
@media (min-width: 769px) and (max-width: 1024px) {
    .block-container {
        padding: 1rem 1rem 1.5rem !important;
    }
    .mef-header h1 {
        font-size: 1.9rem !important;
    }
    .crisis-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    .kpi-grid {
        grid-template-columns: repeat(3, 1fr) !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem !important;
        padding: 0.45rem 0.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER PRINCIPAL
# ─────────────────────────────────────────────

st.markdown("""
<div class="mef-header">
    <div class="mef-header-badge">Feria de Innovación · Tercer Cuatrimestre</div>
    <h1>Simulador Bursátil Cuantitativo</h1>
    <p class="subtitle">Maestría en Economía y Finanzas &nbsp;·&nbsp; Mesa de Dinero Institucional</p>
    <div class="tags">
        <span class="tag">CETES</span>
        <span class="tag">UDIBONOS</span>
        <span class="tag">Convertibles</span>
        <span class="tag">FIBRAS</span>
        <span class="tag">ETFs</span>
        <span class="tag">Black-Scholes</span>
        <span class="tag">Markowitz</span>
        <span class="tag">VaR</span>
        <span class="tag">Bosque Aleatorio</span>
        <span class="tag">Simulador de Crisis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BARRA LATERAL — PORTAFOLIO
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuración del Portafolio")
    st.markdown("---")

    st.markdown("### 💰 Distribución de activos (MXN)")
    portafolio_usuario = {}
    total_portafolio = 0

    activos_config = {
        "CETES":          (200_000, "🏛️"),
        "UDIBONOS":       (150_000, "📈"),
        "Convertibles":   (100_000, "🔄"),
        "FIBRA Uno":      (150_000, "🏢"),
        "FIBRA Danhos":   (100_000, "🏬"),
        "ETF IPC":        (150_000, "📉"),
        "ETF Nasdaq":     (100_000, "💻"),
        "Efectivo/Liquidez": (50_000, "💵"),
    }

    for activo, (default, icono) in activos_config.items():
        val = st.number_input(
            f"{icono} {activo}",
            min_value=0,
            max_value=5_000_000,
            value=default,
            step=10_000,
            format="%d",
        )
        portafolio_usuario[activo] = {"valor": val, "pct": 0}
        total_portafolio += val

    # Calcular porcentajes
    for activo in portafolio_usuario:
        pct = portafolio_usuario[activo]["valor"] / total_portafolio * 100 if total_portafolio > 0 else 0
        portafolio_usuario[activo]["pct"] = round(pct, 1)

    st.markdown(f"### 💼 Total: **${total_portafolio:,.0f} MXN**")
    st.markdown("---")

    st.markdown("### 🌍 Variables Macroeconómicas Base")
    inflacion_base   = st.slider("Inflación actual (%)", 2.0, 25.0, 4.0, 0.5) / 100
    tasa_base        = st.slider("Tasa CETES (%)", 4.0, 20.0, 11.5, 0.25) / 100
    volatilidad_base = st.slider("Volatilidad mercado (%)", 10.0, 60.0, 18.0, 1.0) / 100
    tipo_cambio      = st.slider("Tipo de cambio MXN/USD", 15.0, 25.0, 17.2, 0.1)


# ─────────────────────────────────────────────
#  TABS PRINCIPALES
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏛️ CETES & UDIBONOS",
    "🔄 Convertibles",
    "🚨 Simulador de Crisis",
    "📐 Frontera Eficiente",
    "🤖 IA Predictiva",
    "📋 Portafolio",
])


# ══════════════════════════════════════════════
#  TAB 1 — CETES & UDIBONOS
# ══════════════════════════════════════════════

with tab1:
    st.markdown("## 🏛️ Instrumentos de Deuda Soberana")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Calculadora de CETES")
        vn_cetes    = st.number_input("Valor Nominal (MXN)", value=10.0, step=1.0, key="vn_cetes")
        tasa_cetes  = st.slider("Tasa de rendimiento (%)", 4.0, 20.0, float(tasa_base * 100), 0.1,
                                key="tasa_cetes_calc") / 100
        dias_cetes  = st.selectbox("Plazo", [28, 91, 182, 364], index=3, key="dias_cetes")
        inflacion_c = st.slider("Inflación esperada (%)", 2.0, 30.0, float(inflacion_base * 100), 0.5,
                                key="inf_cetes") / 100

        precio_c = precio_cetes(vn_cetes, tasa_cetes, dias_cetes)
        ganancia = vn_cetes - precio_c
        r_real   = rendimiento_real_cetes(tasa_cetes, inflacion_c)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Precio de compra", f"${precio_c:.4f}")
        with c2:
            st.metric("Ganancia", f"${ganancia:.4f}")
        with c3:
            color_r = "normal" if r_real >= 0 else "inverse"
            st.metric("Rendimiento real", f"{r_real*100:.2f}%",
                      delta=f"{'✓' if r_real >= 0 else '✗'} {'positivo' if r_real >= 0 else 'negativo'}",
                      delta_color=color_r)

        st.markdown("---")
        st.markdown("#### Tabla comparativa por plazo")
        df_tabla = pd.DataFrame(tabla_cetes(vn_cetes, tasa_cetes, inflacion=inflacion_c))
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Ilusión Monetaria — Nominal vs Real")
        tasa_nom_graf = st.slider("Tasa nominal CETES (%)", 4.0, 20.0,
                                   float(tasa_base * 100), 0.5, key="tasa_ilusion") / 100
        horizonte_g = st.slider("Horizonte (años)", 1, 20, 10, 1, key="horizonte_ilusion")

        fig_ilusion = grafica_ilusion_monetaria(
            tasa_nominal=tasa_nom_graf,
            inflacion_escenarios=[0.04, 0.08, 0.15, 0.25],
            horizonte_anos=horizonte_g,
        )
        st.plotly_chart(fig_ilusion, use_container_width=True)

        st.markdown('<div class="concepto-box">'
                    '<b>💡 Ilusión Monetaria (Fisher, 1928):</b> El rendimiento nominal puede '
                    'parecer atractivo, pero si la inflación lo supera, el poder adquisitivo '
                    'real disminuye. Un CETE al 11.5% con inflación del 15% genera una '
                    '<b>pérdida real del 3%</b>.'
                    '</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📈 UDIBONOS — Protección contra Inflación")

    col3, col4 = st.columns(2)
    with col3:
        udi_actual   = st.number_input("Valor UDI actual (MXN)", value=8.15, step=0.01)
        tasa_cupon_r = st.slider("Cupón real UDIBONO (%)", 1.0, 8.0, 4.0, 0.25,
                                  key="cupon_udi") / 100
        tasa_mdo_r   = st.slider("Tasa mercado real (%)", 1.0, 8.0, 3.5, 0.25,
                                  key="tmdo_udi") / 100
        periodos_u   = st.selectbox("Periodos semestrales hasta vencimiento",
                                     [4, 8, 10, 14, 20], index=2, key="per_udi")
        inf_udi      = st.slider("Inflación esperada (%)", 2.0, 30.0,
                                  float(inflacion_base * 100), 0.5, key="inf_udi") / 100

        res_udi = precio_udibono(100, udi_actual, tasa_cupon_r, tasa_mdo_r,
                                  periodos_u, inf_udi)

        for k, v in res_udi.items():
            if k != "Flujos por periodo":
                st.metric(k, v if isinstance(v, str) else
                           (f"${v:,.2f}" if "MXN" in k or "UDI" in k else str(v)))

    with col4:
        comp = comparar_cetes_udibono(tasa_base, tasa_cupon_r, inflacion_base)
        st.markdown("### CETES vs UDIBONO — Comparativa Real")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Rend. real CETES",
                      f"{comp['Rendimiento real CETES (%)']:.2f}%",
                      delta_color="inverse" if comp["Rendimiento real CETES (%)"] < 0 else "normal")
        with m2:
            st.metric("Rend. real UDIBONO",
                      f"{comp['Rendimiento real UDIBONO (%)']:.2f}%")

        ventaja = comp["Ventaja UDIBONO (pp)"]
        if ventaja > 0:
            st.markdown(f'<div class="alerta-verde">✅ {comp["Conclusión"]}<br>'
                        f'El UDIBONO supera al CETE en <b>{ventaja:.2f} pp</b> de rendimiento real.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alerta-roja">⚠️ {comp["Conclusión"]}</div>',
                        unsafe_allow_html=True)

        # VaR
        st.markdown("### 📊 Valor en Riesgo (VaR)")
        conf_var = st.radio("Nivel de confianza", [95, 99], horizontal=True, key="conf_var")
        hor_var  = st.slider("Horizonte (días)", 1, 30, 1, key="hor_var")
        vol_var  = st.slider("Volatilidad diaria (%)", 0.1, 5.0, 0.5, 0.1, key="vol_var") / 100

        res_var = var_parametrico(total_portafolio, vol_var, conf_var / 100, hor_var)
        v1, v2 = st.columns(2)
        with v1:
            st.metric(f"VaR {conf_var}%",
                      f"${res_var[f'VaR {conf_var}% (MXN)']:,.0f} MXN")
        with v2:
            st.metric("VaR relativo",
                      f"{res_var[f'VaR {conf_var}% (%)']:.2f}%")
        st.caption(res_var["Interpretación"])


# ══════════════════════════════════════════════
#  TAB 2 — CONVERTIBLES
# ══════════════════════════════════════════════

with tab2:
    st.markdown("## 🔄 Obligaciones Convertibles — Modelo Black-Scholes")

    st.markdown('<div class="concepto-box">'
                '<b>💡 Concepto Clave:</b> Una obligación convertible es un bono que puede '
                'convertirse en acciones. Su valor = <b>Componente Bono + Componente Opción de Compra</b>. '
                'En una crisis: la acción colapsa pero el <b>bono actúa como airbag financiero</b>.'
                '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Parámetros del Instrumento")
        vn_conv   = st.number_input("Valor Nominal (MXN)", value=10_000, step=1000, key="vn_conv")
        cupon_pct = st.slider("Cupón anual (%)", 2.0, 15.0, 7.0, 0.25, key="cupon_conv") / 100
        per_conv  = st.slider("Periodos anuales", 1, 10, 5, 1, key="per_conv")
        td_conv   = st.slider("Tasa descuento mercado (%)", 4.0, 20.0, 12.0, 0.25, key="td_conv") / 100

        st.markdown("### Parámetros Black-Scholes")
        S_conv    = st.number_input("Precio actual de la acción (MXN)", value=45.0, step=1.0)
        K_conv    = st.number_input("Precio de ejercicio / conversión (MXN)", value=50.0, step=1.0)
        T_conv    = st.slider("Años hasta vencimiento", 0.5, 10.0, 5.0, 0.5, key="T_conv")
        sigma_c   = st.slider("Volatilidad acción (%)", 10.0, 80.0, 30.0, 1.0, key="sig_conv") / 100
        ratio_c   = st.slider("Ratio de conversión (acciones/bono)", 1, 50, 10, 1, key="ratio_conv")

    with col2:
        res_conv = valuar_convertible(
            vn_conv, cupon_pct, per_conv, td_conv,
            S_conv, K_conv, T_conv, tasa_base, sigma_c, ratio_c
        )

        st.markdown("### Valuación del Instrumento")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("💼 Valor Total", f"${res_conv['Valor Total Convertible (MXN)']:,.2f}")
        with m2:
            st.metric("🛡️ Comp. Bono", f"${res_conv['Componente Bono (MXN)']:,.2f}")
        with m3:
            st.metric("📈 Comp. Opción", f"${res_conv['Componente Opción de Compra (MXN)']:,.2f}")

        # Donut de composición
        fig_donut = go.Figure(go.Pie(
            labels=["Componente Bono", "Componente Opción"],
            values=[res_conv["% Bono del Total"], res_conv["% Opción del Total"]],
            hole=0.55,
            marker_colors=["#0d9e6e", "#f4a100"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"{res_conv['Estado'][:10]}",
            showarrow=False, font_size=12, x=0.5, y=0.5
        )
        fig_donut.update_layout(height=280, showlegend=False,
                                 title="Composición del instrumento",
                                 paper_bgcolor="#020617", plot_bgcolor="#0F172A",
                                 font_color="#F8FAFC")
        st.plotly_chart(fig_donut, use_container_width=True)

        estado_color = "alerta-verde" if "En el dinero" in res_conv["Estado"] else "alerta-roja"
        st.markdown(f'<div class="{estado_color}">📌 {res_conv["Estado"]}</div>',
                    unsafe_allow_html=True)

    # Anatomía completa
    st.markdown("---")
    st.markdown("### Anatomía de la Convertible vs Precio de Acción")

    precios_rango = np.linspace(K_conv * 0.3, K_conv * 2.5, 100)
    vals_bono, vals_opcion, vals_total = [], [], []

    for S_test in precios_rango:
        r = valuar_convertible(vn_conv, cupon_pct, per_conv, td_conv,
                                S_test, K_conv, T_conv, tasa_base, sigma_c, ratio_c)
        vals_bono.append(r["Componente Bono (MXN)"])
        vals_opcion.append(r["Componente Opción de Compra (MXN)"])
        vals_total.append(r["Valor Total Convertible (MXN)"])

    fig_anat = grafica_anatomia_convertible(
        precios_rango.tolist(), vals_bono, vals_opcion, vals_total, K_conv
    )
    st.plotly_chart(fig_anat, use_container_width=True)

    # Simulación de crisis sobre la convertible
    st.markdown("---")
    st.markdown("### 🚨 Impacto de Crisis sobre la Convertible")

    col5, col6 = st.columns(2)
    with col5:
        choque_acc = st.slider("Caída de la acción en crisis (%)", -70, -5, -40, 5,
                                key="choque_acc") / 100
        choque_vol = st.slider("Incremento de volatilidad (pp)", 5, 40, 20, 5,
                                key="choque_vol") / 100
        choque_sp  = st.slider("Ampliación del spread crediticio (pp)", 1, 10, 3, 1,
                                key="choque_sp") / 100

    with col6:
        res_cris_conv = sensibilidad_convertible_crisis(
            vn_conv, cupon_pct, per_conv, td_conv,
            S_conv, K_conv, T_conv, tasa_base, sigma_c,
            choque_acc, choque_vol, choque_sp
        )
        st.metric("Valor base", f"${res_cris_conv['Valor BASE (MXN)']:,.2f}")
        st.metric("Valor en crisis", f"${res_cris_conv['Valor CRISIS (MXN)']:,.2f}",
                  delta=f"{res_cris_conv['Caída convertible (%)']:.1f}%",
                  delta_color="inverse")

        st.markdown(f'<div class="alerta-verde">'
                    f'🛡️ <b>Protección del bono: {res_cris_conv["Protección del bono (pp)"]:.1f} pp</b><br>'
                    f'{res_cris_conv["Conclusión"]}'
                    f'</div>', unsafe_allow_html=True)

        # Letras griegas de Black-Scholes
        bs_res = black_scholes(S_conv, K_conv, T_conv, tasa_base, sigma_c, "call")
        st.markdown("#### Letras Griegas del componente opción de compra")
        df_griegas = pd.DataFrame({
            "Letra Griega": ["Delta (Δ)", "Gamma (Γ)", "Theta (Θ) diario", "Vega (ν) por 1% vol", "Rho (ρ) por 1% tasa"],
            "Valor": [bs_res["Delta"], bs_res["Gamma"], bs_res["Theta (diario)"],
                      bs_res["Vega (por 1% vol)"], bs_res["Rho (por 1% tasa)"]],
            "Interpretación": [
                "Sensibilidad del precio de la opción ante $1 en la acción",
                "Cambio en Delta por cada $1 que se mueve la acción",
                "Pérdida de valor por el paso de un día (decaimiento temporal)",
                "Ganancia por cada punto porcentual de aumento en volatilidad",
                "Variación por cada punto porcentual de cambio en la tasa libre de riesgo",
            ],
        })
        st.dataframe(df_griegas, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  TAB 3 — SIMULADOR DE CRISIS
# ══════════════════════════════════════════════

with tab3:
    st.markdown("## 🚨 Simulador de Crisis Bursátiles")
    st.markdown("Selecciona un escenario de crisis para ver su impacto en tiempo real sobre el portafolio.")

    # Botones grandes de crisis
    st.markdown("### Elige tu escenario:")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    crisis_seleccionada = st.session_state.get("crisis_sel", "hiperinflacion")

    with col_b1:
        if st.button("🔥 HIPERINFLACIÓN\nAlza de Tasas", use_container_width=True):
            st.session_state["crisis_sel"] = "hiperinflacion"
            crisis_seleccionada = "hiperinflacion"
    with col_b2:
        if st.button("🦠 PANDEMIA\nConfinamiento", use_container_width=True):
            st.session_state["crisis_sel"] = "pandemia"
            crisis_seleccionada = "pandemia"
    with col_b3:
        if st.button("💥 DEFAULT\nCorporativo", use_container_width=True):
            st.session_state["crisis_sel"] = "default_corporativo"
            crisis_seleccionada = "default_corporativo"
    with col_b4:
        if st.button("⚡ FLASH\nCRASH", use_container_width=True):
            st.session_state["crisis_sel"] = "flash_crash"
            crisis_seleccionada = "flash_crash"

    st.markdown("---")

    # Ejecutar simulación
    resultado = aplicar_crisis(crisis_seleccionada, portafolio={k: v.copy() for k, v in portafolio_usuario.items()})

    cfg_actual = CRISIS_CONFIG[crisis_seleccionada]
    st.markdown(f"## {resultado['Icono']} {resultado['Crisis']}")
    st.markdown(f"*{resultado['Descripción']}*")
    st.markdown(f"📚 **Referencia histórica:** {resultado['Referencia histórica']}")

    # KPIs principales
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Portafolio antes", f"${resultado['Portafolio antes (MXN)']:,.0f}")
    with col_k2:
        st.metric("Portafolio después", f"${resultado['Portafolio después (MXN)']:,.0f}",
                  delta=f"{resultado['Pérdida total (%)']:.1f}%",
                  delta_color="inverse")
    with col_k3:
        st.metric("Pérdida total", f"${abs(resultado['Pérdida total (MXN)']):,.0f} MXN")
    with col_k4:
        st.metric("Activo más resistente", resultado['Activo más resistente'])

    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        fig_impacto = grafica_impacto_crisis(resultado)
        st.plotly_chart(fig_impacto, use_container_width=True)

    with col_g2:
        fig_gauge = grafica_gauge_riesgo(resultado["Pérdida total (%)"], resultado["Crisis"])
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### Cambio en variables macroeconómicas")
        ETIQUETAS_MACRO = {
            "inflacion":     "Inflación",
            "tasa_cetes":    "Tasa CETES",
            "volatilidad":   "Volatilidad",
            "tipo_cambio":   "Tipo de Cambio",
            "spread_credit": "Diferencial Crediticio",
            "liquidez":      "Liquidez",
        }
        macro_a = resultado["Macro antes"]
        macro_d = resultado["Macro después"]
        for var, val_d in macro_d.items():
            delta = round(val_d - macro_a[var], 2)
            etiqueta = ETIQUETAS_MACRO.get(var, var.replace("_", " ").title())
            if delta != 0:
                st.metric(etiqueta, f"{val_d:.2f}%",
                          delta=f"{delta:+.2f}pp",
                          delta_color="inverse" if delta > 0 and var in ["inflacion", "volatilidad", "spread_credit"] else "normal")

    # Narrativas por activo
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <h3>Narrativa por Activo</h3>
    </div>""", unsafe_allow_html=True)
    cols_nar = st.columns(4)
    activos_list = list(resultado["Detalle por activo"].keys())
    for i, activo in enumerate(activos_list):
        det = resultado["Detalle por activo"][activo]
        col_idx = i % 4
        es_positivo = det["Impacto (%)"] >= 0
        clase = "activo-positivo" if es_positivo else "activo-negativo"
        signo = "+" if es_positivo else ""
        with cols_nar[col_idx]:
            st.markdown(
                f'<div class="activo-card {clase}">'
                f'<div class="ac-nombre">{activo}</div>'
                f'<div class="ac-impacto">{signo}{det["Impacto (%)"]:.1f}%</div>'
                f'<div class="ac-valor">${det["Valor después (MXN)"]:,.0f} MXN</div>'
                f'<div class="ac-texto">{det["Narrativa"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Evolución temporal
    st.markdown("---")
    st.markdown("### 📈 Evolución temporal del portafolio")
    dias_sim = st.slider("Días de simulación", 60, 252, 120, 10, key="dias_evo")

    evo = evolucion_temporal_crisis(crisis_seleccionada, dias_sim,
                                     {k: v.copy() for k, v in portafolio_usuario.items()})
    fig_evo = grafica_evolucion_portafolio(evo)
    st.plotly_chart(fig_evo, use_container_width=True)

    # Conceptos educativos del Flash Crash
    if crisis_seleccionada == "flash_crash" and resultado.get("Concepto educativo"):
        st.markdown("---")
        st.markdown("### 🎓 Conceptos de Innovación Financiera")
        conceptos = resultado["Concepto educativo"]
        c1, c2 = st.columns(2)
        for i, (concepto, explicacion) in enumerate(conceptos.items()):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(f'<div class="concepto-box"><b>🔬 {concepto}:</b><br>{explicacion}</div>',
                            unsafe_allow_html=True)

    # Comparativa de las 4 crisis
    st.markdown("---")
    st.markdown("### 🗺️ Mapa Comparativo — Las 4 Crisis")
    comparativa = comparar_todas_las_crisis({k: v.copy() for k, v in portafolio_usuario.items()})
    fig_comp = grafica_comparativa_crisis(comparativa)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown('<div class="concepto-box">'
                '💡 <b>Lectura del mapa:</b> Verde = activo sube o resiste bien | '
                'Rojo = activo cae fuertemente. '
                'Los CETES y UDIBONOS tienden a ser defensivos; las FIBRAS y ETFs '
                'son los más sensibles a shocks de mercado.'
                '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 4 — FRONTERA EFICIENTE
# ══════════════════════════════════════════════

with tab4:
    st.markdown("## 📐 Frontera Eficiente de Markowitz")
    st.markdown('<div class="concepto-box">'
                '💡 <b>Teoría Moderna del Portafolio (Markowitz, 1952):</b> '
                'Para un nivel de riesgo dado, existe una composición óptima de activos '
                'que maximiza el rendimiento esperado. La frontera eficiente representa ese '
                'conjunto de portafolios óptimos. El <b>Índice de Sharpe</b> = (Rend. - Tasa libre) / Riesgo'
                'mide el rendimiento por unidad de riesgo.'
                '</div>', unsafe_allow_html=True)

    st.markdown("### Rendimientos esperados anuales por activo")
    col_fe1, col_fe2 = st.columns([1, 2])

    with col_fe1:
        activos_fe = ["CETES", "UDIBONOS", "Convertibles", "FIBRA", "ETF IPC", "ETF Nasdaq"]
        rend_def = [0.115, 0.085, 0.12, 0.10, 0.14, 0.16]
        vol_def  = [0.003, 0.005, 0.20, 0.18, 0.22, 0.20]

        rendimientos_fe = []
        volatilidades_fe = []

        for i, (act, rd, vd) in enumerate(zip(activos_fe, rend_def, vol_def)):
            r = st.slider(f"Rend. esperado {act} (%)", 2.0, 30.0, rd * 100, 0.5,
                          key=f"rend_{act}") / 100
            rendimientos_fe.append(r)

        for i, (act, vd) in enumerate(zip(activos_fe, vol_def)):
            v = st.slider(f"Volatilidad {act} (%)", 0.1, 50.0, vd * 100, 0.5,
                          key=f"vol_{act}") / 100
            volatilidades_fe.append(v)

        n_port = st.slider("Portafolios a simular", 1000, 10000, 5000, 500, key="n_port")

    with col_fe2:
        # Construir matriz de covarianza con correlaciones supuestas
        n_act = len(activos_fe)
        # Correlaciones calibradas para activos mexicanos
        corr_matrix = np.array([
            [1.00, 0.70, 0.20, 0.10, 0.05, 0.02],
            [0.70, 1.00, 0.25, 0.15, 0.08, 0.04],
            [0.20, 0.25, 1.00, 0.55, 0.60, 0.50],
            [0.10, 0.15, 0.55, 1.00, 0.65, 0.40],
            [0.05, 0.08, 0.60, 0.65, 1.00, 0.70],
            [0.02, 0.04, 0.50, 0.40, 0.70, 1.00],
        ])
        vols = np.array(volatilidades_fe)
        cov_matrix = np.outer(vols, vols) * corr_matrix

        with st.spinner("Simulando 5,000 portafolios..."):
            res_fe = frontera_eficiente(
                rendimientos_fe, cov_matrix, activos_fe, n_port, tasa_base
            )

        fig_fe = grafica_frontera_eficiente(res_fe)
        st.plotly_chart(fig_fe, use_container_width=True)

        # Mostrar pesos del portafolio óptimo
        p_opt = res_fe["portafolio_max_sharpe"]
        p_min = res_fe["portafolio_min_varianza"]

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.markdown("#### ⭐ Portafolio de Máximo Índice Sharpe")
            st.metric("Rendimiento", f"{p_opt['Rendimiento (%)']:.2f}%")
            st.metric("Riesgo", f"{p_opt['Riesgo (%)']:.2f}%")
            st.metric("Índice de Sharpe", f"{p_opt['Índice de Sharpe']:.4f}")
            pesos_df = pd.DataFrame(list(p_opt["Pesos"].items()),
                                     columns=["Activo", "Peso (%)"])
            st.dataframe(pesos_df, use_container_width=True, hide_index=True)

        with col_opt2:
            st.markdown("#### 🛡️ Portafolio de Mínima Varianza")
            st.metric("Rendimiento", f"{p_min['Rendimiento (%)']:.2f}%")
            st.metric("Riesgo", f"{p_min['Riesgo (%)']:.2f}%")
            st.metric("Índice de Sharpe", f"{p_min['Índice de Sharpe']:.4f}")
            pesos_df2 = pd.DataFrame(list(p_min["Pesos"].items()),
                                      columns=["Activo", "Peso (%)"])
            st.dataframe(pesos_df2, use_container_width=True, hide_index=True)

        # Mapa de correlaciones
        st.markdown("#### Mapa de Correlaciones")
        fig_corr = grafica_correlaciones(corr_matrix, activos_fe)
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 — IA PREDICTIVA
# ══════════════════════════════════════════════

with tab5:
    st.markdown("## 🤖 IA Predictiva — Bosque Aleatorio")
    st.markdown('<div class="concepto-box">'
                '🤖 <b>Aprendizaje Automático aplicado a Finanzas:</b> Un Bosque Aleatorio entrena '
                '200 árboles de decisión sobre datos históricos de escenarios macroeconómicos '
                'para predecir qué clase de activo será la más resiliente. '
                '<b>No es magia: es estadística multivariada aplicada.</b>'
                '</div>', unsafe_allow_html=True)

    col_ia1, col_ia2 = st.columns([1, 1])

    with col_ia1:
        st.markdown("### 🎛️ Condiciones del Entorno Macro")
        inf_ia   = st.slider("Inflación (%)", 2.0, 30.0, float(inflacion_base * 100), 0.5,
                              key="inf_ia") / 100
        tasa_ia  = st.slider("Tasa CETES (%)", 4.0, 20.0, float(tasa_base * 100), 0.25,
                              key="tasa_ia") / 100
        vol_ia   = st.slider("Volatilidad (%)", 10.0, 70.0, float(volatilidad_base * 100), 1.0,
                              key="vol_ia") / 100
        tc_ia    = st.slider("Tipo de cambio MXN/USD", 15.0, 25.0, tipo_cambio, 0.1, key="tc_ia")
        sp_ia    = st.slider("Spread crediticio (%)", 0.5, 10.0, 2.0, 0.25, key="sp_ia") / 100
        liq_ia   = st.slider("Liquidez (índice)", 0.3, 1.3, 0.9, 0.05, key="liq_ia")
        pib_ia   = st.slider("Crecimiento PIB (%)", -8.0, 6.0, 1.5, 0.5, key="pib_ia") / 100
        conf_ia  = st.slider("Confianza consumidor", 60, 130, 90, 5, key="conf_ia")

        predecir = st.button("🔮 Predecir Activo Más Resiliente", use_container_width=True)

    with col_ia2:
        if predecir or "ia_resultado" in st.session_state:
            with st.spinner("Entrenando Bosque Aleatorio con 800 escenarios históricos..."):
                # Almacenar modelo Y encoder en session_state por usuario (evita estado global compartido)
                if "ia_modelo_obj" not in st.session_state:
                    metricas, modelo_obj, encoder_obj = entrenar_modelo(800)
                    st.session_state["ia_modelo"]      = metricas
                    st.session_state["ia_modelo_obj"]  = modelo_obj
                    st.session_state["ia_encoder_obj"] = encoder_obj

                resultado_ia = predecir_activo_resiliente(
                    inf_ia, tasa_ia, vol_ia, tc_ia, sp_ia, liq_ia, pib_ia, conf_ia,
                    modelo=st.session_state["ia_modelo_obj"],
                    encoder=st.session_state["ia_encoder_obj"],
                )
                st.session_state["ia_resultado"] = resultado_ia

            resultado_ia = st.session_state["ia_resultado"]
            metricas = st.session_state.get("ia_modelo", {})

            st.markdown(f"### 🏆 Activo Recomendado")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a73e8,#0d9e6e);color:white;
                        padding:1.5rem;border-radius:12px;text-align:center;margin:0.5rem 0">
                <div style="font-size:2.5rem;font-weight:700">{resultado_ia['Activo recomendado']}</div>
                <div style="font-size:1.1rem;opacity:0.9">Confianza: {resultado_ia['Confianza del modelo (%)']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="concepto-box">🧠 {resultado_ia["Razonamiento"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="concepto-box">📊 {resultado_ia["Interpretación"]}</div>',
                        unsafe_allow_html=True)

            # Top 3
            st.markdown("#### Top 3 candidatos")
            for rango, (act, prob) in enumerate(resultado_ia["Top 3 candidatos"]):
                medal = ["🥇", "🥈", "🥉"][rango]
                st.progress(int(prob), text=f"{medal} {act}: {prob:.1f}%")

            # Métricas del modelo
            if metricas:
                st.markdown("---")
                st.markdown(f"#### Métricas del modelo")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Precisión", f"{metricas.get('Precisión (Accuracy)', 0):.1f}%")
                with m2:
                    st.metric("Escenarios entrenamiento", metricas.get('Escenarios de entrenamiento', 0))
                with m3:
                    st.metric("Árboles", metricas.get('Árboles en el bosque', 0))

    # Importancia de variables
    st.markdown("---")
    st.markdown("### 📊 Importancia de Variables en el Modelo")

    if "ia_modelo_obj" in st.session_state:
        df_imp = importancia_variables_df(modelo=st.session_state["ia_modelo_obj"])
        fig_imp = px.bar(
            df_imp, x="Importancia (%)", y="Variable",
            orientation="h", color="Importancia (%)",
            color_continuous_scale="Blues",
            title="¿Qué variables considera más el modelo para decidir?",
        )
        fig_imp.update_layout(plot_bgcolor="#0F172A", paper_bgcolor="#020617",
                               font_color="#F8FAFC", height=380,
                               showlegend=False, coloraxis_showscale=False)
        fig_imp.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown('<div class="concepto-box">'
                    '💡 Las variables de mayor importancia son las que el modelo usa '
                    'con más frecuencia para separar escenarios. La inflación y la tasa '
                    'CETES suelen dominar en economías emergentes como México.'
                    '</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Ejecuta la predicción primero para ver la importancia de variables.")


# ══════════════════════════════════════════════
#  TAB 6 — PORTAFOLIO ACTUAL
# ══════════════════════════════════════════════

with tab6:
    st.markdown("## 📋 Estado Actual del Portafolio")

    # Pie chart de distribución
    activos_names = list(portafolio_usuario.keys())
    valores_names = [portafolio_usuario[a]["valor"] for a in activos_names]
    colores_pie   = [
        "#1a73e8", "#0d9e6e", "#f4a100", "#9c27b0",
        "#7b1fa2", "#e53935", "#ef6c00", "#607d8b"
    ]

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        fig_pie = go.Figure(go.Pie(
            labels=activos_names,
            values=valores_names,
            hole=0.45,
            marker_colors=colores_pie,
            textinfo="label+percent",
            hovertemplate="%{label}<br>$%{value:,.0f} MXN<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"${total_portafolio:,.0f}",
            showarrow=False, font_size=14, font_color="#F8FAFC",
        )
        fig_pie.update_layout(
            title="Distribución del Portafolio",
            height=400, paper_bgcolor="#020617", plot_bgcolor="#0F172A",
            font_color="#F8FAFC",
            legend=dict(orientation="v", x=1.0, y=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_p2:
        st.markdown("### Resumen del Portafolio")
        df_port = pd.DataFrame([
            {
                "Activo": a,
                "Valor (MXN)": f"${portafolio_usuario[a]['valor']:,.0f}",
                "Peso (%)": f"{portafolio_usuario[a]['pct']:.1f}%",
            }
            for a in activos_names
            if portafolio_usuario[a]["valor"] > 0
        ])
        st.dataframe(df_port, use_container_width=True, hide_index=True)

        st.metric("💼 Total del Portafolio", f"${total_portafolio:,.0f} MXN")

        # Clasificación por riesgo
        pct_rf = sum(portafolio_usuario[a]["pct"] for a in ["CETES", "UDIBONOS", "Efectivo/Liquidez"])
        pct_rv = 100 - pct_rf

        st.markdown("#### Perfil de Riesgo")
        st.progress(int(pct_rf), text=f"Renta Fija / Defensivos: {pct_rf:.1f}%")
        st.progress(int(pct_rv), text=f"Renta Variable / Crecimiento: {pct_rv:.1f}%")

        perfil = (
            "🛡️ Conservador" if pct_rf > 60 else
            "⚖️ Moderado"    if pct_rf > 35 else
            "🚀 Agresivo"
        )
        st.markdown(f"**Perfil de inversor estimado: {perfil}**")

        # Duración del portafolio de renta fija
        st.markdown("#### Duración del Portafolio de Renta Fija")
        flujos_rf = [10_000 * 0.115] * 3 + [10_000 * 1.115]  # Simplificado
        dur_mac = duracion_macaulay(flujos_rf, tasa_base)
        dur_mod = duracion_modificada(flujos_rf, tasa_base)
        conv_v  = convexidad(flujos_rf, tasa_base)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Duración Macaulay", f"{dur_mac:.2f} años")
        with d2:
            st.metric("Duración Modificada", f"{dur_mod:.4f}")
        with d3:
            st.metric("Convexidad", f"{conv_v:.4f}")

        # Sensibilidad ante alza de 100pb
        sens = cambio_precio_bono(
            sum(portafolio_usuario[a]["valor"] for a in ["CETES", "UDIBONOS"]),
            dur_mod, conv_v, 0.01
        )
        st.markdown('<div class="alerta-roja">'
                    f'⚠️ Ante un alza de 100 pb en tasas, la cartera de renta fija '
                    f'perdería aproximadamente <b>{abs(sens["Cambio estimado (%)"]):.2f}%</b> '
                    f'= <b>${abs(sens["Precio original"] - sens["Nuevo precio estimado"]):,.0f} MXN</b>'
                    '</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<div class="mef-footer">
    <strong>Simulador Bursátil Cuantitativo</strong> — Maestría en Economía y Finanzas<br>
    Modelos: Black-Scholes &nbsp;·&nbsp; Markowitz &nbsp;·&nbsp; Fisher &nbsp;·&nbsp; VaR &nbsp;·&nbsp; Bosque Aleatorio<br>
    <span>Parámetros calibrados con mercados reales MX &nbsp;·&nbsp; Python 3 &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; scikit-learn</span>
</div>
""", unsafe_allow_html=True)
