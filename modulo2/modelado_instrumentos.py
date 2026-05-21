"""
MÓDULO 2 — Modelado de Instrumentos Financieros
=================================================
Cubre:
  - Obligaciones Convertibles (Black-Scholes + componente bono)
  - FIBRAS (REITs mexicanos): beta, correlación, dividend yield
  - ETFs: tracking error, beta de mercado
  - Frontera Eficiente simplificada (Markowitz)

Maestría en Economía y Finanzas — Feria de Innovación
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# ─────────────────────────────────────────────
#  MODELO BLACK-SCHOLES
# ─────────────────────────────────────────────

def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, tipo: str = "call") -> dict:
    """
    Precio de opción europea usando Black-Scholes (1973).

    Parámetros
    ----------
    S     : Precio actual de la acción subyacente (MXN)
    K     : Precio de ejercicio / strike (MXN)
    T     : Tiempo hasta vencimiento en años (ej. 0.5 = 6 meses)
    r     : Tasa libre de riesgo anual continua (ej. 0.115)
    sigma : Volatilidad implícita anual (ej. 0.30 = 30%)
    tipo  : "call" o "put"

    Retorna
    -------
    dict con precio, griegas (delta, gamma, theta, vega, rho)
    """
    # Validación de inputs para evitar NaN silencioso
    if S <= 0:
        raise ValueError(f"Precio de la acción S={S} debe ser > 0")
    if K <= 0:
        raise ValueError(f"Strike K={K} debe ser > 0")
    if sigma <= 0:
        raise ValueError(f"Volatilidad sigma={sigma} debe ser > 0")

    if T <= 0:
        # Al vencimiento
        if tipo == "call":
            precio = max(S - K, 0)
        else:
            precio = max(K - S, 0)
        return {"Precio": precio, "Delta": 1 if S > K else 0,
                "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == "call":
        precio = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta  = norm.cdf(d1)
        rho    = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        precio = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta  = norm.cdf(d1) - 1
        rho    = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma  = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta  = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
               - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo == "call" else -d2)) / 365
    vega   = S * norm.pdf(d1) * np.sqrt(T) / 100

    nombre_tipo = "Opción de Compra (Call)" if tipo == "call" else "Opción de Venta (Put)"
    return {
        "Tipo": nombre_tipo,
        "Precio": round(precio, 4),
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 6),
        "Theta (diario)": round(theta, 4),
        "Vega (por 1% vol)": round(vega, 4),
        "Rho (por 1% tasa)": round(rho, 4),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
    }


# ─────────────────────────────────────────────
#  OBLIGACIONES CONVERTIBLES
# ─────────────────────────────────────────────

def valuar_convertible(
    valor_nominal: float,
    tasa_cupon: float,
    periodos: int,
    tasa_descuento: float,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    ratio_conversion: float = 1.0,
) -> dict:
    """
    Valúa una Obligación Convertible como:
        Valor = Componente Bono + Componente Opción Call

    Concepto clave: cuando ocurre una crisis, la parte accionaria se destruye
    pero el componente bono ofrece protección parcial al tenedor.

    Parámetros
    ----------
    valor_nominal    : VN del bono (MXN)
    tasa_cupon       : Cupón anual (ej. 0.08)
    periodos         : Número de periodos hasta vencimiento
    tasa_descuento   : Tasa de descuento del mercado crediticio
    S                : Precio actual de la acción
    K                : Strike de conversión
    T                : Tiempo hasta vencimiento (años)
    r                : Tasa libre de riesgo
    sigma            : Volatilidad de la acción
    ratio_conversion : Acciones por bono (delta de conversión)
    """
    # Componente bono: valor presente de cupones + principal
    cupon = valor_nominal * tasa_cupon
    pv_cupones = sum(cupon / (1 + tasa_descuento) ** i
                     for i in range(1, periodos + 1))
    pv_principal = valor_nominal / (1 + tasa_descuento) ** periodos
    valor_bono = pv_cupones + pv_principal

    # Componente opción (call europeo por cada acción convertible)
    bs = black_scholes(S, K, T, r, sigma, "call")
    valor_opcion = bs["Precio"] * ratio_conversion

    valor_total = valor_bono + valor_opcion

    # Floor: precio mínimo = valor del bono puro (protección ante caída accionaria)
    floor_proteccion = valor_bono

    return {
        "Valor Total Convertible (MXN)": round(valor_total, 2),
        "Componente Bono (MXN)": round(valor_bono, 2),
        "Componente Opción de Compra (MXN)": round(valor_opcion, 2),
        "Piso de Protección (MXN)": round(floor_proteccion, 2),
        "Prima de Conversión (MXN)": round(valor_opcion, 2),
        "% Bono del Total": round(valor_bono / valor_total * 100, 1),
        "% Opción del Total": round(valor_opcion / valor_total * 100, 1),
        "Precio Acción": S,
        "Precio de Ejercicio": K,
        "Volatilidad (%)": round(sigma * 100, 1),
        "Estado": "En el dinero — conviene convertir" if S > K else "Fuera del dinero — conviene mantener como bono",
    }


def sensibilidad_convertible_crisis(
    valor_nominal: float, tasa_cupon: float, periodos: int,
    tasa_descuento_base: float, S_base: float, K: float,
    T: float, r: float, sigma_base: float,
    choque_accion: float = -0.40,
    choque_volatilidad: float = 0.20,
    choque_spread: float = 0.03,
) -> dict:
    """
    Simula el impacto de una crisis sobre una convertible.
    Muestra cómo el componente bono amortigua la caída accionaria.
    """
    # Valuación base
    base = valuar_convertible(valor_nominal, tasa_cupon, periodos,
                               tasa_descuento_base, S_base, K, T, r, sigma_base)

    # Valuación bajo crisis
    S_crisis = S_base * (1 + choque_accion)
    sigma_crisis = sigma_base + choque_volatilidad
    tasa_crisis = tasa_descuento_base + choque_spread

    crisis = valuar_convertible(valor_nominal, tasa_cupon, periodos,
                                 tasa_crisis, S_crisis, K, T, r, sigma_crisis)

    caida_accion_pura = choque_accion * 100
    caida_convertible = (crisis["Valor Total Convertible (MXN)"] /
                          base["Valor Total Convertible (MXN)"] - 1) * 100

    return {
        "Valor BASE (MXN)": base["Valor Total Convertible (MXN)"],
        "Valor CRISIS (MXN)": crisis["Valor Total Convertible (MXN)"],
        "Caída acción pura (%)": round(caida_accion_pura, 1),
        "Caída convertible (%)": round(caida_convertible, 1),
        "Protección del bono (pp)": round(caida_accion_pura - caida_convertible, 1),
        "Componente bono crisis (MXN)": crisis["Componente Bono (MXN)"],
        "Componente opción crisis (MXN)": crisis["Componente Opción de Compra (MXN)"],
        "Conclusión": (
            f"El componente bono absorbió {round(caida_accion_pura - caida_convertible, 1)} pp "
            f"de la caída. El instrumento cayó solo {round(abs(caida_convertible), 1)}% "
            f"vs {round(abs(caida_accion_pura), 1)}% de la acción sola."
        ),
    }


# ─────────────────────────────────────────────
#  FIBRAS (REITs Mexicanos)
# ─────────────────────────────────────────────

def analizar_fibra(
    retornos_fibra: np.ndarray,
    retornos_mercado: np.ndarray,
    precio_actual: float,
    dividendo_anual: float,
    nombre: str = "FIBRA",
) -> dict:
    """
    Analiza una FIBRA usando regresión lineal (CAPM).

    Métricas calculadas:
    - Beta (sensibilidad al mercado)
    - Alpha de Jensen (rendimiento excedente ajustado por riesgo)
    - Correlación con el mercado
    - Dividend Yield
    - Sharpe Ratio
    """
    retornos_fibra = np.array(retornos_fibra)
    retornos_mercado = np.array(retornos_mercado)

    # Beta = Cov(r_fibra, r_mercado) / Var(r_mercado)
    cov_matrix = np.cov(retornos_fibra, retornos_mercado)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Correlación de Pearson
    correlacion = np.corrcoef(retornos_fibra, retornos_mercado)[0, 1]

    # Alpha de Jensen: r_fibra - [rf + beta * (r_mercado - rf)]
    rf_diaria = 0.115 / 252  # Tasa libre de riesgo diaria aprox.
    alpha = (np.mean(retornos_fibra) - rf_diaria
             - beta * (np.mean(retornos_mercado) - rf_diaria)) * 252

    # Volatilidad anualizada
    volatilidad = np.std(retornos_fibra) * np.sqrt(252)

    # Sharpe Ratio
    rf_anual = 0.115
    sharpe = (np.mean(retornos_fibra) * 252 - rf_anual) / volatilidad

    # Dividend Yield
    div_yield = dividendo_anual / precio_actual

    return {
        "Instrumento": nombre,
        "Beta": round(beta, 4),
        "Alpha anual (%)": round(alpha * 100, 2),
        "Correlación con mercado": round(correlacion, 4),
        "Volatilidad anual (%)": round(volatilidad * 100, 2),
        "Índice de Sharpe": round(sharpe, 4),
        "Rendimiento por Dividendo (%)": round(div_yield * 100, 2),
        "Rendimiento Total esperado (%)": round((np.mean(retornos_fibra) * 252 + div_yield) * 100, 2),
        "Interpretación Beta": (
            f"Beta={round(beta, 2)}: la FIBRA se mueve {round(abs(beta), 2)}x el mercado. "
            + ("Más volátil que el mercado." if abs(beta) > 1 else "Más defensiva que el mercado.")
        ),
    }


# ─────────────────────────────────────────────
#  ETFs
# ─────────────────────────────────────────────

def analizar_etf(
    retornos_etf: np.ndarray,
    retornos_indice: np.ndarray,
    nombre_etf: str = "ETF",
    nombre_indice: str = "Índice de referencia",
) -> dict:
    """
    Evalúa la calidad de replicación de un ETF vs su índice de referencia.

    Tracking Error: desviación estándar de la diferencia de retornos.
    Un TE bajo indica mejor replicación.
    """
    retornos_etf    = np.array(retornos_etf)
    retornos_indice = np.array(retornos_indice)

    diferencias = retornos_etf - retornos_indice
    tracking_error = np.std(diferencias) * np.sqrt(252)  # Anualizado

    # Beta del ETF vs su índice (idealmente debe ser ≈ 1.0)
    cov = np.cov(retornos_etf, retornos_indice)
    beta = cov[0, 1] / cov[1, 1]
    correlacion = np.corrcoef(retornos_etf, retornos_indice)[0, 1]

    rendimiento_etf    = np.mean(retornos_etf) * 252
    rendimiento_indice = np.mean(retornos_indice) * 252
    diferencia_anual   = rendimiento_etf - rendimiento_indice

    return {
        "ETF": nombre_etf,
        "Índice": nombre_indice,
        "Error de Seguimiento anual (%)": round(tracking_error * 100, 3),
        "Beta vs Índice": round(beta, 4),
        "Correlación": round(correlacion, 4),
        "Rendimiento ETF anual (%)": round(rendimiento_etf * 100, 2),
        "Rendimiento Índice anual (%)": round(rendimiento_indice * 100, 2),
        "Diferencia anual (pp)": round(diferencia_anual * 100, 3),
        "Calidad replicación": (
            "Excelente (TE < 0.5%)" if tracking_error < 0.005 else
            "Buena (TE < 1%)"       if tracking_error < 0.01  else
            "Aceptable (TE < 2%)"   if tracking_error < 0.02  else
            "Deficiente (TE > 2%)"
        ),
    }


# ─────────────────────────────────────────────
#  FRONTERA EFICIENTE (Markowitz)
# ─────────────────────────────────────────────

def frontera_eficiente(
    retornos_esperados: list,
    matriz_covarianza: np.ndarray,
    nombres_activos: list,
    n_portafolios: int = 5000,
    tasa_libre_riesgo: float = 0.115,
) -> dict:
    """
    Simulación de Monte Carlo de portafolios aleatorios para
    trazar la Frontera Eficiente de Markowitz.

    Retorna coordenadas (riesgo, rendimiento) y el portafolio
    óptimo de máximo Sharpe Ratio.
    """
    n = len(retornos_esperados)
    retornos_esperados = np.array(retornos_esperados)

    resultados = np.zeros((3, n_portafolios))
    pesos_todos = np.zeros((n_portafolios, n))

    for i in range(n_portafolios):
        pesos = np.random.dirichlet(np.ones(n))
        pesos_todos[i] = pesos

        ret_port  = np.dot(pesos, retornos_esperados)
        riesgo_port = np.sqrt(pesos @ matriz_covarianza @ pesos)
        sharpe_port = (ret_port - tasa_libre_riesgo) / riesgo_port

        resultados[0, i] = riesgo_port
        resultados[1, i] = ret_port
        resultados[2, i] = sharpe_port

    # Portafolio de máximo Sharpe
    idx_max_sharpe = np.argmax(resultados[2])
    pesos_max_sharpe = pesos_todos[idx_max_sharpe]

    # Portafolio de mínima varianza
    idx_min_var = np.argmin(resultados[0])
    pesos_min_var = pesos_todos[idx_min_var]

    return {
        "riesgos": resultados[0].tolist(),
        "rendimientos": resultados[1].tolist(),
        "sharpes": resultados[2].tolist(),
        "portafolio_max_sharpe": {
            "Rendimiento (%)": round(resultados[1, idx_max_sharpe] * 100, 2),
            "Riesgo (%)": round(resultados[0, idx_max_sharpe] * 100, 2),
            "Índice de Sharpe": round(resultados[2, idx_max_sharpe], 4),
            "Pesos": {nombres_activos[j]: round(pesos_max_sharpe[j] * 100, 1)
                      for j in range(n)},
        },
        "portafolio_min_varianza": {
            "Rendimiento (%)": round(resultados[1, idx_min_var] * 100, 2),
            "Riesgo (%)": round(resultados[0, idx_min_var] * 100, 2),
            "Índice de Sharpe": round(resultados[2, idx_min_var], 4),
            "Pesos": {nombres_activos[j]: round(pesos_min_var[j] * 100, 1)
                      for j in range(n)},
        },
        "n_portafolios_simulados": n_portafolios,
        "activos": nombres_activos,
    }
