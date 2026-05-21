"""
MÓDULO 3 — Simulador de Crisis Bursátiles
==========================================
Modela 4 escenarios de crisis con parámetros calibrados
con base en eventos históricos reales (México y global):

  BOTÓN 1 — Hiperinflación y Alza de Tasas (México 1994-95, 2022)
  BOTÓN 2 — Pandemia / Confinamiento (COVID-19, 2020)
  BOTÓN 3 — Default Corporativo (Vitro, Cablevisión, LatAm)
  BOTÓN 4 — Flash Crash (Mayo 2010, Agosto 2015)

Cada crisis modifica variables macroeconómicas y calcula
el impacto en cada clase de activo del portafolio.

Maestría en Economía y Finanzas — Feria de Innovación
"""

import numpy as np


# ─────────────────────────────────────────────
#  PORTAFOLIO BASE POR DEFECTO
# ─────────────────────────────────────────────

PORTAFOLIO_BASE = {
    "CETES":         {"valor": 200_000, "pct": 20.0},
    "UDIBONOS":      {"valor": 150_000, "pct": 15.0},
    "Convertibles":  {"valor": 100_000, "pct": 10.0},
    "FIBRA Uno":     {"valor": 150_000, "pct": 15.0},
    "FIBRA Danhos":  {"valor": 100_000, "pct": 10.0},
    "ETF IPC":       {"valor": 150_000, "pct": 15.0},
    "ETF Nasdaq":    {"valor": 100_000, "pct": 10.0},
    "Efectivo/Liquidez": {"valor": 50_000, "pct": 5.0},
}

MACRO_BASE = {
    "inflacion":    0.04,   # 4 % anual
    "tasa_cetes":   0.115,  # 11.5 % (referencia Banxico)
    "volatilidad":  0.18,   # Volatilidad implícita promedio
    "tipo_cambio":  17.20,  # MXN/USD
    "spread_credit":0.012,  # 120 pb spread corporativo
    "liquidez":     1.0,    # Índice de liquidez (1 = normal)
}


# ─────────────────────────────────────────────
#  DEFINICIÓN DE CHOQUES POR CRISIS
# ─────────────────────────────────────────────

CRISIS_CONFIG = {
    "hiperinflacion": {
        "nombre": "Hiperinflación y Alza de Tasas",
        "icono": "🔥",
        "descripcion": (
            "Escenario de presión inflacionaria severa con respuesta agresiva "
            "de política monetaria. Basado en crisis 1994-95 (Efecto Tequila) "
            "y el ciclo de alzas 2021-2023."
        ),
        "referencia_historica": "México 1994-95: inflación 51.7%, tasas 80%+",
        "choques_macro": {
            "inflacion":     +0.15,
            "tasa_cetes":    +0.08,
            "volatilidad":   +0.20,
            "tipo_cambio":   +0.25,
            "spread_credit": +0.02,
            "liquidez":      -0.20,
        },
        "impacto_activos": {
            "CETES":         -0.12,   # Pierden valor real aunque nominalmente positivos
            "UDIBONOS":      +0.08,   # Protegen: están ligados a inflación
            "Convertibles":  -0.18,   # Tasa alta destruye valor bono + acción volátil
            "FIBRA Uno":     -0.22,   # Alta tasa encarece financiamiento inmobiliario
            "FIBRA Danhos":  -0.20,
            "ETF IPC":       -0.30,   # Mercado cae ante incertidumbre
            "ETF Nasdaq":    -0.15,   # Diversificación internacional amortigua
            "Efectivo/Liquidez": +0.02,
        },
        "narrativa": {
            "CETES": "El CETE ofrece tasa nominal alta, pero la inflación del 15%+ erosiona el poder real.",
            "UDIBONOS": "El UDIBONO brilla en este escenario: su principal crece con la inflación (UDIS).",
            "Convertibles": "Doble golpe: tasas altas bajan el componente bono, volatilidad sube el call pero incertidumbre domina.",
            "FIBRA Uno": "Las FIBRAS sufren: financiamiento más caro, valor de propiedades presionado.",
            "ETF IPC": "El IPC cae ante fuga de capitales y aversión al riesgo.",
            "ETF Nasdaq": "El Nasdaq en dólares ofrece refugio parcial si el MXN se deprecia.",
        },
    },

    "pandemia": {
        "nombre": "Pandemia / Confinamiento",
        "icono": "🦠",
        "descripcion": (
            "Choque exógeno de demanda con cierre de actividades económicas. "
            "Basado en COVID-19 (marzo–abril 2020): caída súbita, "
            "recuperación en V para mercados financieros."
        ),
        "referencia_historica": "COVID-19 (Mar 2020): IPC -35%, FIBRA Uno -45%, CETES +0.5%",
        "choques_macro": {
            "inflacion":     -0.01,
            "tasa_cetes":    -0.05,   # Banxico recortó tasas de emergencia
            "volatilidad":   +0.35,   # VIX pasó de 14 a 85 puntos
            "tipo_cambio":   +0.20,   # Depreciación del MXN
            "spread_credit": +0.04,
            "liquidez":      -0.40,   # Iliquidez extrema en mercados
        },
        "impacto_activos": {
            "CETES":         +0.03,   # Refugio de corto plazo, tasas bajaron
            "UDIBONOS":      -0.05,   # Deflación inicial; recuperación posterior
            "Convertibles":  -0.25,   # Colapso accionario, bono como colchón parcial
            "FIBRA Uno":     -0.45,   # Cierre comercial: oficinas y plazas vacías
            "FIBRA Danhos":  -0.48,   # Danhos muy expuesto a centros comerciales
            "ETF IPC":       -0.35,
            "ETF Nasdaq":    -0.20,   # Tech sufrió menos (Zoom, Netflix...)
            "Efectivo/Liquidez": +0.01,
        },
        "narrativa": {
            "CETES": "Los CETES actúan como refugio: el gobierno no defaultea y la tasa bajó.",
            "UDIBONOS": "Deflación inicial presiona al UDIBONO; pero su protección vuelve con el rebote inflacionario.",
            "Convertibles": "Las acciones colapsan; el componente bono actúa como 'airbag' financiero.",
            "FIBRA Uno": "Colapso total: oficinas vacías, rentas sin pagar, desocupación récord.",
            "ETF Nasdaq": "Las empresas tecnológicas se benefician del trabajo remoto vs sectores tradicionales.",
        },
    },

    "default_corporativo": {
        "nombre": "Default Corporativo",
        "icono": "💥",
        "descripcion": (
            "Quiebra de un emisor corporativo importante con contagio crediticio. "
            "Basado en Vitro (2011), Cablevisión MX (2020), Mexichem/Orbia. "
            "Mide riesgo de crédito y riesgo sistémico."
        ),
        "referencia_historica": "Vitro 2010-11: mayor reestructura corporativa en México",
        "choques_macro": {
            "inflacion":     +0.01,
            "tasa_cetes":    +0.01,
            "volatilidad":   +0.25,
            "tipo_cambio":   +0.08,
            "spread_credit": +0.06,   # Spread corporativo explota
            "liquidez":      -0.30,
        },
        "impacto_activos": {
            "CETES":         +0.02,   # Refugio soberano; no hay riesgo de crédito
            "UDIBONOS":      +0.01,
            "Convertibles":  -0.55,   # Si la emisora defaultea, el convertible vale casi cero
            "FIBRA Uno":     -0.15,   # Contagio crediticio afecta REITs
            "FIBRA Danhos":  -0.18,
            "ETF IPC":       -0.20,   # Pérdida de confianza sistémica
            "ETF Nasdaq":    -0.08,   # Menor exposición al riesgo crediticio MX
            "Efectivo/Liquidez": +0.01,
        },
        "narrativa": {
            "CETES": "El soberano mexicano (Gobierno Federal) no defaultea: los CETES son refugio.",
            "Convertibles": "Si el emisor quiebra, la convertible se convierte en crédito quirografario en concurso mercantil.",
            "FIBRA Uno": "El contagio crediticio eleva los spreads de todo el sector real estate.",
            "ETF IPC": "El mercado descuenta riesgo sistémico; el IPC cae por aversión al riesgo.",
        },
    },

    "flash_crash": {
        "nombre": "Flash Crash",
        "icono": "⚡",
        "descripcion": (
            "Caída súbita y recuperación parcial causada por trading algorítmico "
            "de alta frecuencia. Liquidez desaparece en minutos. "
            "Basado en Flash Crash del 6 de mayo de 2010 (Dow Jones -1,000 pts en minutos) "
            "y Agosto 2015 (caída China)."
        ),
        "referencia_historica": "Flash Crash 6 May 2010: S&P 500 -9.2% en 36 minutos, recuperó en 1 hora",
        "choques_macro": {
            "inflacion":     0.00,
            "tasa_cetes":    0.00,
            "volatilidad":   +0.50,   # VIX explota instantáneamente
            "tipo_cambio":   +0.05,
            "spread_credit": +0.03,
            "liquidez":      -0.70,   # Mercado sin compradores
        },
        "impacto_activos": {
            "CETES":         +0.01,   # Refugio, aunque no en tiempo real (mercado OTC)
            "UDIBONOS":      +0.01,
            "Convertibles":  -0.30,   # Acción cae; opción pierde theta
            "FIBRA Uno":     -0.28,   # Sin liquidez, spread bid-ask explota
            "FIBRA Danhos":  -0.30,
            "ETF IPC":       -0.38,   # ETFs sufren más: precio de mercado vs NAV diverge
            "ETF Nasdaq":    -0.35,
            "Efectivo/Liquidez": +0.00,
        },
        "narrativa": {
            "CETES": "Mercado OTC: los CETES no se transan en bolsa, se protegen del derrumbe relámpago.",
            "ETF IPC": "El ETF cotiza en bolsa: cuando no hay compradores, el precio cae más que el valor real del fondo (VNA).",
            "Convertibles": "La opción pierde valor por el salto de volatilidad; el bono no se mueve igual de rápido.",
            "FIBRA Uno": "Sin formadores de mercado, el diferencial compra-venta de las FIBRAS se dispara; precio de mercado colapsa.",
        },
        "concepto_educativo": {
            "Negociación de Alta Frecuencia": "Algoritmos ejecutan miles de órdenes por segundo (HFT: High-Frequency Trading).",
            "Formación de Mercado": "Los operadores de alta frecuencia proveen liquidez en condiciones normales; en crisis la retiran instantáneamente.",
            "Error de Seguimiento Extremo": "Durante un derrumbe relámpago, el ETF puede cotizar con 5-10% de descuento a su valor real (VNA).",
            "Disyuntores de Mercado": "Mecanismo que detiene la negociación si el mercado cae más del 7% en el día.",
        },
    },
}


# ─────────────────────────────────────────────
#  MOTOR DE SIMULACIÓN
# ─────────────────────────────────────────────

def aplicar_crisis(
    tipo_crisis: str,
    portafolio: dict = None,
    macro_base: dict = None,
) -> dict:
    """
    Aplica los choques de una crisis a un portafolio dado.

    Parámetros
    ----------
    tipo_crisis : str  → "hiperinflacion" | "pandemia" | "default_corporativo" | "flash_crash"
    portafolio  : dict → Diccionario activo → {"valor": float, "pct": float}
    macro_base  : dict → Variables macroeconómicas iniciales

    Retorna
    -------
    dict con estado antes, estado después, impactos por activo y métricas de riesgo.
    """
    if portafolio is None:
        portafolio = {k: v.copy() for k, v in PORTAFOLIO_BASE.items()}
    if macro_base is None:
        macro_base = MACRO_BASE.copy()

    config = CRISIS_CONFIG.get(tipo_crisis)
    if config is None:
        raise ValueError(f"Crisis '{tipo_crisis}' no definida. Opciones: {list(CRISIS_CONFIG.keys())}")

    # Calcular macro post-crisis
    macro_crisis = {}
    for k, v in macro_base.items():
        choque = config["choques_macro"].get(k, 0)
        macro_crisis[k] = round(v + choque, 4)

    # Calcular impacto en cada activo
    valor_total_antes = sum(a["valor"] for a in portafolio.values())
    resultados_activos = {}
    valor_total_despues = 0

    for activo, datos in portafolio.items():
        impacto = config["impacto_activos"].get(activo, 0)
        valor_antes = datos["valor"]
        valor_despues = valor_antes * (1 + impacto)
        ganancia_perdida = valor_despues - valor_antes

        resultados_activos[activo] = {
            "Valor antes (MXN)": round(valor_antes, 2),
            "Impacto (%)": round(impacto * 100, 1),
            "Valor después (MXN)": round(valor_despues, 2),
            "Ganancia/Pérdida (MXN)": round(ganancia_perdida, 2),
            "Narrativa": config["narrativa"].get(activo, ""),
        }
        valor_total_despues += valor_despues

    perdida_total = valor_total_despues - valor_total_antes
    perdida_pct = perdida_total / valor_total_antes * 100

    # Activo más afectado y más resistente
    impactos = {k: config["impacto_activos"].get(k, 0) for k in portafolio}
    mas_afectado   = min(impactos, key=impactos.get)
    mas_resistente = max(impactos, key=impactos.get)

    return {
        "Crisis": config["nombre"],
        "Icono": config["icono"],
        "Descripción": config["descripcion"],
        "Referencia histórica": config["referencia_historica"],

        "Macro antes": {k: round(v * 100, 2) for k, v in macro_base.items()},
        "Macro después": {k: round(v * 100, 2) for k, v in macro_crisis.items()},

        "Portafolio antes (MXN)": round(valor_total_antes, 2),
        "Portafolio después (MXN)": round(valor_total_despues, 2),
        "Pérdida total (MXN)": round(perdida_total, 2),
        "Pérdida total (%)": round(perdida_pct, 2),

        "Activo más afectado": mas_afectado,
        "Activo más resistente": mas_resistente,

        "Detalle por activo": resultados_activos,

        "Concepto educativo": config.get("concepto_educativo", {}),
    }


def comparar_todas_las_crisis(portafolio: dict = None) -> dict:
    """
    Ejecuta los 4 escenarios de crisis y genera tabla comparativa.
    Útil para mostrar en la feria cómo cada activo reacciona distinto.
    """
    if portafolio is None:
        portafolio = {k: v.copy() for k, v in PORTAFOLIO_BASE.items()}

    resultados = {}
    for crisis_key in CRISIS_CONFIG:
        r = aplicar_crisis(crisis_key, portafolio={k: v.copy() for k, v in portafolio.items()})
        resultados[crisis_key] = {
            "Nombre": r["Crisis"],
            "Pérdida portafolio (%)": r["Pérdida total (%)"],
            "Impactos": {activo: r["Detalle por activo"][activo]["Impacto (%)"]
                         for activo in portafolio},
        }
    return resultados


def evolucion_temporal_crisis(
    tipo_crisis: str,
    dias: int = 90,
    portafolio: dict = None,
) -> dict:
    """
    Simula la evolución día a día de un portafolio durante y después de una crisis.
    Usa Movimiento Browniano Geométrico (GBM) con parámetros modificados por la crisis.

    Retorna series de tiempo para graficar.
    """
    if portafolio is None:
        portafolio = {k: v.copy() for k, v in PORTAFOLIO_BASE.items()}

    config = CRISIS_CONFIG.get(tipo_crisis)
    if config is None:
        raise ValueError(f"Crisis '{tipo_crisis}' no definida. Opciones: {list(CRISIS_CONFIG.keys())}")

    sigma_crisis = MACRO_BASE["volatilidad"] + config["choques_macro"]["volatilidad"]

    # RNG aislado: no contamina el estado global de numpy
    rng = np.random.default_rng(seed=42)
    series = {}
    dias_array = np.arange(dias)

    dia_inicio_crisis = int(dias * 0.3)  # La crisis inicia al 30% del horizonte
    dia_recuperacion  = int(dias * 0.6)  # Recuperación parcial desde 60%

    for activo, datos in portafolio.items():
        impacto_total = config["impacto_activos"].get(activo, 0)
        # Volatilidad diferenciada por activo
        vol_activo = {
            "CETES": 0.003, "UDIBONOS": 0.005, "Convertibles": 0.025,
            "FIBRA Uno": 0.018, "FIBRA Danhos": 0.020,
            "ETF IPC": 0.022, "ETF Nasdaq": 0.020,
            "Efectivo/Liquidez": 0.001,
        }.get(activo, 0.015)

        precio = [datos["valor"]]
        for d in range(1, dias):
            if d < dia_inicio_crisis:
                drift = 0.0001
                shock = rng.normal(0, vol_activo)
            elif d < dia_recuperacion:
                fraccion = (d - dia_inicio_crisis) / (dia_recuperacion - dia_inicio_crisis)
                drift = (impacto_total * fraccion) / (dia_recuperacion - dia_inicio_crisis)
                shock = rng.normal(0, vol_activo * (1 + sigma_crisis))
            else:
                drift = abs(impacto_total) * 0.002
                shock = rng.normal(0, vol_activo * 0.8)

            nuevo_precio = precio[-1] * (1 + drift + shock)
            precio.append(max(nuevo_precio, 0))

        series[activo] = [round(p, 2) for p in precio]

    return {
        "crisis": tipo_crisis,
        "nombre_crisis": config["nombre"],
        "dias": dias_array.tolist(),
        "series": series,
        "dia_inicio_crisis": dia_inicio_crisis,
        "dia_recuperacion": dia_recuperacion,
    }
