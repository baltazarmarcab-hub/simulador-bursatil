"""
MÓDULO 5 — Visualización Científica
=====================================
Funciones de graficación con Plotly para:
  - Evolución del portafolio bajo crisis
  - Frontera Eficiente de Markowitz
  - Rendimiento nominal vs real (ilusión monetaria)
  - Mapa de calor de correlaciones
  - Anatomía de una convertible (bono vs opción)
  - Comparativa de crisis por activo

Maestría en Economía y Finanzas — Feria de Innovación
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


# Paleta de colores institucional
COLORES_ACTIVOS = {
    "CETES":              "#1a73e8",
    "UDIBONOS":           "#0d9e6e",
    "Convertibles":       "#f4a100",
    "FIBRA Uno":          "#9c27b0",
    "FIBRA Danhos":       "#7b1fa2",
    "ETF IPC":            "#e53935",
    "ETF Nasdaq":         "#ef6c00",
    "Efectivo/Liquidez":  "#607d8b",
}

COLORES_CRISIS = {
    "hiperinflacion":      "#e53935",
    "pandemia":            "#7b1fa2",
    "default_corporativo": "#f57c00",
    "flash_crash":         "#0288d1",
}


# ─────────────────────────────────────────────
#  1. EVOLUCIÓN DEL PORTAFOLIO
# ─────────────────────────────────────────────

def grafica_evolucion_portafolio(datos_crisis: dict) -> go.Figure:
    """
    Grafica la evolución temporal de cada activo durante la crisis.
    Marca el inicio de la crisis y el inicio de recuperación.
    """
    series = datos_crisis["series"]
    dias   = datos_crisis["dias"]
    nombre = datos_crisis["nombre_crisis"]
    d_crisis = datos_crisis["dia_inicio_crisis"]
    d_recup  = datos_crisis["dia_recuperacion"]

    fig = go.Figure()

    for activo, valores in series.items():
        # Normalizar a 100 para comparar en la misma escala
        base = valores[0] if valores[0] != 0 else 1
        norm = [v / base * 100 for v in valores]

        fig.add_trace(go.Scatter(
            x=dias, y=norm,
            name=activo,
            line=dict(color=COLORES_ACTIVOS.get(activo, "#888"), width=2),
            hovertemplate=f"<b>{activo}</b><br>Día %{{x}}<br>Índice: %{{y:.1f}}<extra></extra>",
        ))

    # Línea base
    fig.add_hline(y=100, line_dash="dot", line_color="gray",
                  annotation_text="Valor inicial = 100")

    # Zona de crisis
    fig.add_vrect(x0=d_crisis, x1=d_recup,
                  fillcolor="rgba(229,57,53,0.1)",
                  layer="below", line_width=0,
                  annotation_text="Crisis activa",
                  annotation_position="top left")

    # Marca inicio crisis
    fig.add_vline(x=d_crisis, line_dash="dash", line_color="#e53935",
                  annotation_text="Inicio crisis")
    fig.add_vline(x=d_recup, line_dash="dash", line_color="#43a047",
                  annotation_text="Inicio recuperación")

    fig.update_layout(
        title=dict(text=f"Evolución del Portafolio — {nombre}", font=dict(size=20)),
        xaxis_title="Días",
        yaxis_title="Valor indexado (base 100)",
        legend=dict(orientation="v", x=1.01, y=1),
        hovermode="x unified",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=500,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


# ─────────────────────────────────────────────
#  2. IMPACTO POR ACTIVO (BARRAS)
# ─────────────────────────────────────────────

def grafica_impacto_crisis(resultado_crisis: dict) -> go.Figure:
    """
    Gráfica de barras horizontales con el impacto de cada activo bajo la crisis seleccionada.
    """
    detalle = resultado_crisis["Detalle por activo"]
    activos  = list(detalle.keys())
    impactos = [detalle[a]["Impacto (%)"] for a in activos]
    colores  = ["#43a047" if v >= 0 else "#e53935" for v in impactos]

    fig = go.Figure(go.Bar(
        x=impactos, y=activos,
        orientation="h",
        marker_color=colores,
        text=[f"{v:+.1f}%" for v in impactos],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Impacto: %{x:.1f}%<extra></extra>",
    ))

    fig.add_vline(x=0, line_color="black", line_width=1)

    fig.update_layout(
        title=dict(
            text=f"{resultado_crisis['Icono']} Impacto por Activo — {resultado_crisis['Crisis']}",
            font=dict(size=18)
        ),
        xaxis_title="Cambio en valor (%)",
        yaxis_title="",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=420,
    )
    return fig


# ─────────────────────────────────────────────
#  3. FRONTERA EFICIENTE
# ─────────────────────────────────────────────

def grafica_frontera_eficiente(datos_frontera: dict) -> go.Figure:
    """
    Scatter plot de portafolios simulados (Monte Carlo) con
    coloración por Sharpe Ratio y marcadores especiales para
    portafolio de máximo Sharpe y mínima varianza.
    """
    riesgos      = [r * 100 for r in datos_frontera["riesgos"]]
    rendimientos = [r * 100 for r in datos_frontera["rendimientos"]]
    sharpes      = datos_frontera["sharpes"]

    p_max = datos_frontera["portafolio_max_sharpe"]
    p_min = datos_frontera["portafolio_min_varianza"]

    fig = go.Figure()

    # Nube de portafolios
    fig.add_trace(go.Scatter(
        x=riesgos, y=rendimientos,
        mode="markers",
        marker=dict(
            color=sharpes,
            colorscale="RdYlGn",
            size=4,
            opacity=0.6,
            colorbar=dict(title="Índice de Sharpe"),
        ),
        name="Portafolios simulados",
        hovertemplate="Riesgo: %{x:.2f}%<br>Rendimiento: %{y:.2f}%<extra></extra>",
    ))

    # Portafolio máximo Sharpe
    fig.add_trace(go.Scatter(
        x=[p_max["Riesgo (%)"]],
        y=[p_max["Rendimiento (%)"]],
        mode="markers+text",
        marker=dict(color="#1a73e8", size=16, symbol="star"),
        text=["⭐ Máx Índice Sharpe"],
        textposition="top right",
        name=f"Máx Índice Sharpe ({p_max['Índice de Sharpe']:.2f})",
        hovertemplate=(
            "<b>Portafolio Óptimo</b><br>"
            f"Rendimiento: {p_max['Rendimiento (%)']}%<br>"
            f"Riesgo: {p_max['Riesgo (%)']}%<br>"
            f"Índice de Sharpe: {p_max['Índice de Sharpe']}"
            "<extra></extra>"
        ),
    ))

    # Portafolio mínima varianza
    fig.add_trace(go.Scatter(
        x=[p_min["Riesgo (%)"]],
        y=[p_min["Rendimiento (%)"]],
        mode="markers+text",
        marker=dict(color="#e53935", size=14, symbol="diamond"),
        text=["🛡 Mín Riesgo"],
        textposition="top right",
        name="Mínima Varianza",
        hovertemplate=(
            "<b>Mínima Varianza</b><br>"
            f"Rendimiento: {p_min['Rendimiento (%)']}%<br>"
            f"Riesgo: {p_min['Riesgo (%)']}%"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Frontera Eficiente de Markowitz — Simulación Monte Carlo", font=dict(size=18)),
        xaxis_title="Riesgo — Desviación Estándar (%)",
        yaxis_title="Rendimiento Esperado Anual (%)",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=500,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ─────────────────────────────────────────────
#  4. INFLACIÓN REAL vs NOMINAL (Ilusión Monetaria)
# ─────────────────────────────────────────────

def grafica_ilusion_monetaria(
    tasa_nominal: float = 0.115,
    inflacion_escenarios: list = None,
    horizonte_anos: int = 10,
) -> go.Figure:
    """
    Muestra cómo $100 MXN crecen en términos nominales vs reales
    bajo distintos niveles de inflación. Demuestra la ilusión monetaria.
    """
    if inflacion_escenarios is None:
        inflacion_escenarios = [0.04, 0.08, 0.15, 0.25]

    anos = list(range(horizonte_anos + 1))
    fig = go.Figure()

    # Crecimiento nominal siempre igual
    nominal = [100 * (1 + tasa_nominal) ** t for t in anos]
    fig.add_trace(go.Scatter(
        x=anos, y=nominal,
        name=f"Nominal ({tasa_nominal*100:.1f}%)",
        line=dict(color="#1a73e8", width=3, dash="solid"),
        hovertemplate="Año %{x}: $%{y:.2f} nominales<extra></extra>",
    ))

    colores_inflacion = ["#43a047", "#f4a100", "#e53935", "#7b1fa2"]
    for i, inf in enumerate(inflacion_escenarios):
        tasa_real = (1 + tasa_nominal) / (1 + inf) - 1
        real = [100 * (1 + tasa_real) ** t for t in anos]

        sufre = " ✗" if tasa_real < 0 else " ✓"
        fig.add_trace(go.Scatter(
            x=anos, y=real,
            name=f"Real con inflación {inf*100:.0f}%{sufre}",
            line=dict(color=colores_inflacion[i], width=2, dash="dot"),
            hovertemplate=f"Año %{{x}}: $%{{y:.2f}} reales (inf={inf*100:.0f}%)<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="Poder adquisitivo original")

    fig.update_layout(
        title=dict(
            text=f"Ilusión Monetaria — CETES {tasa_nominal*100:.1f}% nominal vs distintas inflaciones",
            font=dict(size=17)
        ),
        xaxis_title="Años",
        yaxis_title="Valor de $100 MXN invertidos",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=460,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ─────────────────────────────────────────────
#  5. ANATOMÍA DE LA CONVERTIBLE
# ─────────────────────────────────────────────

def grafica_anatomia_convertible(
    precio_acciones: list,
    valores_bono: list,
    valores_opcion: list,
    valores_total: list,
    strike: float,
) -> go.Figure:
    """
    Muestra la composición de una convertible (bono + opción call)
    en función del precio de la acción subyacente.
    Permite ver cuándo actúa como bono y cuándo como acción.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Valor de la Convertible vs Precio de Acción",
                        "Composición: % Bono vs % Opción"),
        vertical_spacing=0.15,
    )

    # Gráfica superior: valores absolutos
    fig.add_trace(go.Scatter(
        x=precio_acciones, y=valores_total,
        name="Convertible Total",
        line=dict(color="#1a73e8", width=3),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=precio_acciones, y=valores_bono,
        name="Componente Bono (piso de protección)",
        line=dict(color="#0d9e6e", width=2, dash="dot"),
        fill="tonexty" if False else None,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=precio_acciones, y=valores_opcion,
        name="Componente Opción de Compra",
        line=dict(color="#f4a100", width=2, dash="dash"),
    ), row=1, col=1)

    # Línea de conversión
    fig.add_vline(x=strike, line_dash="dash", line_color="#e53935",
                  annotation_text=f"Precio de ejercicio: ${strike:.0f}",
                  row=1, col=1)

    # Anotaciones de región
    fig.add_annotation(
        x=strike * 0.6, y=max(valores_bono) * 0.9,
        text="🛡 Zona Bono<br>(Fuera del dinero)",
        showarrow=False,
        bgcolor="rgba(13,158,110,0.15)", bordercolor="#0d9e6e",
        font=dict(size=11, color="#F8FAFC"), row=1, col=1
    )
    fig.add_annotation(
        x=strike * 1.4, y=max(valores_total) * 0.85,
        text="📈 Zona Acción<br>(En el dinero)",
        showarrow=False,
        bgcolor="rgba(244,161,0,0.15)", bordercolor="#f4a100",
        font=dict(size=11), row=1, col=1
    )

    # Gráfica inferior: composición porcentual
    pct_bono   = [b / t * 100 if t > 0 else 100 for b, t in zip(valores_bono, valores_total)]
    pct_opcion = [o / t * 100 if t > 0 else 0   for o, t in zip(valores_opcion, valores_total)]

    fig.add_trace(go.Scatter(
        x=precio_acciones, y=pct_bono,
        name="% Bono",
        line=dict(color="#0d9e6e"),
        fill="tozeroy",
        fillcolor="rgba(13,158,110,0.15)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=precio_acciones, y=pct_opcion,
        name="% Opción",
        line=dict(color="#f4a100"),
        fill="tonexty",
        fillcolor="rgba(244,161,0,0.15)",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text="Anatomía de la Obligación Convertible", font=dict(size=18)),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=620,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Precio de la Acción (MXN)", row=1, col=1)
    fig.update_xaxes(title_text="Precio de la Acción (MXN)", row=2, col=1)
    fig.update_yaxes(title_text="Valor (MXN)", row=1, col=1)
    fig.update_yaxes(title_text="Composición (%)", row=2, col=1)
    return fig


# ─────────────────────────────────────────────
#  6. MAPA DE CALOR DE CORRELACIONES
# ─────────────────────────────────────────────

def grafica_correlaciones(matriz_corr: np.ndarray, nombres: list) -> go.Figure:
    """
    Heatmap de correlaciones entre activos del portafolio.
    Verde = baja correlación (diversificación), rojo = alta correlación.
    """
    fig = go.Figure(go.Heatmap(
        z=matriz_corr,
        x=nombres,
        y=nombres,
        colorscale="RdYlGn_r",
        zmid=0,
        text=np.round(matriz_corr, 2),
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Correlación"),
    ))

    fig.update_layout(
        title=dict(text="Mapa de Correlaciones del Portafolio", font=dict(size=18)),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=480,
        xaxis=dict(tickangle=-45),
    )
    return fig


# ─────────────────────────────────────────────
#  7. COMPARATIVA DE LAS 4 CRISIS
# ─────────────────────────────────────────────

def grafica_comparativa_crisis(datos_comparativa: dict) -> go.Figure:
    """
    Heatmap que compara el impacto de cada crisis sobre cada activo.
    Permite ver de un vistazo qué protege y qué destruye cada escenario.
    """
    crisis_nombres = [datos_comparativa[k]["Nombre"] for k in datos_comparativa]
    activos = list(list(datos_comparativa.values())[0]["Impactos"].keys())

    matriz = []
    for k in datos_comparativa:
        fila = [datos_comparativa[k]["Impactos"][a] for a in activos]
        matriz.append(fila)

    # Trasponer: activos en filas, crisis en columnas
    matriz_t = np.array(matriz).T.tolist()

    fig = go.Figure(go.Heatmap(
        z=matriz_t,
        x=crisis_nombres,
        y=activos,
        colorscale="RdYlGn",
        zmid=0,
        zmin=-0.6,
        zmax=0.15,
        text=[[f"{v:+.0f}%" for v in fila] for fila in matriz_t],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Impacto (%)"),
    ))

    fig.update_layout(
        title=dict(
            text="Comparativa de Crisis — Impacto por Activo (%)",
            font=dict(size=18)
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#020617",
        height=480,
        xaxis=dict(tickangle=-15, title="Escenario de Crisis"),
        yaxis=dict(title="Activo"),
    )
    return fig


# ─────────────────────────────────────────────
#  8. GAUGE DE RIESGO DEL PORTAFOLIO
# ─────────────────────────────────────────────

def grafica_gauge_riesgo(perdida_pct: float, nombre_crisis: str) -> go.Figure:
    """
    Gauge (velocímetro) que muestra el nivel de riesgo del portafolio.
    """
    valor_abs = abs(perdida_pct)
    color = "#43a047" if valor_abs < 10 else "#f4a100" if valor_abs < 25 else "#e53935"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=valor_abs,
        title={"text": f"Pérdida del Portafolio<br>{nombre_crisis}", "font": {"size": 16}},
        delta={"reference": 10, "increasing": {"color": "#e53935"}, "decreasing": {"color": "#43a047"}},
        gauge={
            "axis": {"range": [0, 60], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 10],  "color": "#e8f5e9"},
                {"range": [10, 25], "color": "#fff3e0"},
                {"range": [25, 60], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#b71c1c", "width": 3},
                "thickness": 0.75,
                "value": valor_abs,
            },
        },
        number={"suffix": "%", "font": {"size": 36}},
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor="#020617",
        font=dict(color="#F8FAFC"),
    )
    return fig
