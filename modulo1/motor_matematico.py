"""
MÓDULO 1 — Motor Matemático
============================
Contiene los modelos cuantitativos base para:
  - CETES (Certificados de la Tesorería)
  - UDIBONOS (Bonos ligados a inflación / UDIS)
  - Duración y Convexidad de bonos
  - Valor en Riesgo (VaR) paramétrico

Maestría en Economía y Finanzas — Feria de Innovación
"""

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────
#  CETES
# ─────────────────────────────────────────────

def precio_cetes(valor_nominal: float, tasa_anual: float, dias: int) -> float:
    """
    Precio de mercado de un CETE usando descuento simple.

    Fórmula oficial Banxico:
        P = VN / (1 + r * t)
    donde t = días / 360

    Parámetros
    ----------
    valor_nominal : float  → Valor de redención (generalmente 10 MXN)
    tasa_anual    : float  → Tasa de rendimiento anual (ej. 0.115 = 11.5 %)
    dias          : int    → Plazo en días (28, 91, 182, 364)

    Retorna
    -------
    float → Precio de compra (menor que valor_nominal)
    """
    t = dias / 360
    return valor_nominal / (1 + tasa_anual * t)


def rendimiento_cetes(precio: float, valor_nominal: float, dias: int) -> float:
    """
    Calcula la tasa de rendimiento implícita dado un precio de mercado.

        r = (VN/P - 1) / t
    """
    t = dias / 360
    return (valor_nominal / precio - 1) / t


def rendimiento_real_cetes(tasa_nominal: float, inflacion: float) -> float:
    """
    Rendimiento real usando la ecuación de Fisher:
        (1 + r_real) = (1 + r_nominal) / (1 + inflacion)
    """
    return (1 + tasa_nominal) / (1 + inflacion) - 1


def tabla_cetes(valor_nominal: float, tasa_anual: float,
                plazos: list = None, inflacion: float = 0.04) -> list:
    """
    Genera tabla comparativa de CETES para distintos plazos.
    Retorna lista de dicts con precio, tasa nominal y tasa real.
    """
    if plazos is None:
        plazos = [28, 91, 182, 364]
    resultado = []
    for d in plazos:
        precio = precio_cetes(valor_nominal, tasa_anual, d)
        r_real = rendimiento_real_cetes(tasa_anual, inflacion)
        resultado.append({
            "Plazo (días)": d,
            "Precio MXN": round(precio, 4),
            "Tasa Nominal (%)": round(tasa_anual * 100, 2),
            "Tasa Real (%)": round(r_real * 100, 2),
            "Ganancia MXN": round(valor_nominal - precio, 4),
        })
    return resultado


# ─────────────────────────────────────────────
#  UDIBONOS
# ─────────────────────────────────────────────

def valor_udi(udi_base: float, inflacion_anual: float, dias: int) -> float:
    """
    Proyecta el valor de la UDI a futuro.

    La UDI se actualiza diariamente con base en el INPC.
    Aproximación continua:
        UDI(t) = UDI_0 * (1 + inflacion_anual)^(dias/365)
    """
    return udi_base * (1 + inflacion_anual) ** (dias / 365)


def precio_udibono(
    valor_nominal_udis: float,
    udi_actual: float,
    tasa_cupon_real: float,
    tasa_mercado_real: float,
    periodos: int,
    inflacion_anual: float,
    frecuencia: int = 2,
) -> dict:
    """
    Precio de un UDIBONO en pesos nominales.

    Los UDIBONOS pagan cupón semestral en UDIS; el principal también está
    en UDIS, protegiendo al tenedor contra la inflación.

    Parámetros
    ----------
    valor_nominal_udis  : float → VN expresado en UDIS (generalmente 100)
    udi_actual          : float → Valor vigente de la UDI en MXN
    tasa_cupon_real     : float → Cupón real anual (ej. 0.04 = 4 %)
    tasa_mercado_real   : float → Tasa real de mercado (para descontar)
    periodos            : int   → Número de periodos hasta vencimiento
    inflacion_anual     : float → Inflación esperada (para proyectar UDIS)
    frecuencia          : int   → Pagos por año (2 = semestral)

    Retorna dict con precio en UDIS, precio en MXN y flujos.
    """
    cupon_udi = valor_nominal_udis * (tasa_cupon_real / frecuencia)
    tasa_periodo = tasa_mercado_real / frecuencia

    # Valor presente de cupones en UDIS
    flujos = []
    pv_cupones = 0.0
    for i in range(1, periodos + 1):
        pv = cupon_udi / (1 + tasa_periodo) ** i
        flujos.append(round(pv, 4))
        pv_cupones += pv

    # Valor presente del principal en UDIS
    pv_principal = valor_nominal_udis / (1 + tasa_periodo) ** periodos

    precio_udis = pv_cupones + pv_principal
    precio_mxn = precio_udis * udi_actual

    return {
        "Precio en UDIS": round(precio_udis, 4),
        "Precio en MXN": round(precio_mxn, 2),
        "PV Cupones (UDIS)": round(pv_cupones, 4),
        "PV Principal (UDIS)": round(pv_principal, 4),
        "Flujos por periodo": flujos,
        "UDI utilizada": udi_actual,
    }


def comparar_cetes_udibono(
    tasa_cetes: float,
    tasa_cupon_real: float,
    inflacion: float,
    horizonte_dias: int = 364,
) -> dict:
    """
    Compara el rendimiento real de CETES vs UDIBONO para el mismo horizonte.
    Demuestra la 'ilusión monetaria' cuando la inflación es alta.
    """
    r_real_cetes = rendimiento_real_cetes(tasa_cetes, inflacion)
    # UDIBONO protege; su rendimiento real es aproximadamente la tasa cupón
    r_real_udibono = tasa_cupon_real

    return {
        "Inflación esperada (%)": round(inflacion * 100, 2),
        "Tasa nominal CETES (%)": round(tasa_cetes * 100, 2),
        "Rendimiento real CETES (%)": round(r_real_cetes * 100, 2),
        "Rendimiento real UDIBONO (%)": round(r_real_udibono * 100, 2),
        "Ventaja UDIBONO (pp)": round((r_real_udibono - r_real_cetes) * 100, 2),
        "Conclusión": (
            "UDIBONO protege mejor el poder adquisitivo"
            if r_real_udibono > r_real_cetes
            else "CETES ofrece mejor rendimiento real en este escenario"
        ),
    }


# ─────────────────────────────────────────────
#  DURACIÓN Y CONVEXIDAD
# ─────────────────────────────────────────────

def duracion_macaulay(flujos: list, tasa: float, frecuencia: int = 1) -> float:
    """
    Duración de Macaulay: promedio ponderado del tiempo de cobro de flujos.
    Mide la sensibilidad al precio ante cambios en tasa de interés.

        D = Σ [t * PV(flujo_t)] / Precio
    """
    precio = sum(f / (1 + tasa / frecuencia) ** (i + 1)
                 for i, f in enumerate(flujos))
    if precio == 0:
        return 0
    duracion = sum(
        ((i + 1) / frecuencia) * (f / (1 + tasa / frecuencia) ** (i + 1))
        for i, f in enumerate(flujos)
    ) / precio
    return round(duracion, 4)


def duracion_modificada(flujos: list, tasa: float, frecuencia: int = 1) -> float:
    """
    Duración Modificada: aproxima el % de cambio en precio por cada +1% en tasa.
        D_mod = D_Macaulay / (1 + tasa/frecuencia)
    """
    d_mac = duracion_macaulay(flujos, tasa, frecuencia)
    return round(d_mac / (1 + tasa / frecuencia), 4)


def convexidad(flujos: list, tasa: float, frecuencia: int = 1) -> float:
    """
    Convexidad: corrección de segundo orden para cambios grandes en tasa.
    Mejora la aproximación de la duración modificada.
    """
    precio = sum(f / (1 + tasa / frecuencia) ** (i + 1)
                 for i, f in enumerate(flujos))
    if precio == 0:
        return 0
    conv = sum(
        ((i + 1) * (i + 2)) / (frecuencia ** 2) *
        (f / (1 + tasa / frecuencia) ** (i + 3))
        for i, f in enumerate(flujos)
    ) / precio
    return round(conv, 4)


def cambio_precio_bono(precio_actual: float, dur_mod: float,
                       conv: float, delta_tasa: float) -> dict:
    """
    Aproxima el nuevo precio de un bono ante un choque de tasa:
        ΔP/P ≈ -D_mod * Δr + ½ * Convexidad * Δr²
    """
    cambio_pct = -dur_mod * delta_tasa + 0.5 * conv * delta_tasa ** 2
    nuevo_precio = precio_actual * (1 + cambio_pct)
    return {
        "Precio original": round(precio_actual, 2),
        "Δ Tasa (pp)": round(delta_tasa * 100, 2),
        "Cambio estimado (%)": round(cambio_pct * 100, 2),
        "Nuevo precio estimado": round(nuevo_precio, 2),
    }


# ─────────────────────────────────────────────
#  VALOR EN RIESGO (VaR)
# ─────────────────────────────────────────────

def var_parametrico(valor_portafolio: float, volatilidad_diaria: float,
                    confianza: float = 0.95, horizonte_dias: int = 1) -> dict:
    """
    VaR paramétrico (método varianza-covarianza).

    Supone rendimientos normalmente distribuidos.
        VaR = V * σ_d * √h * z_α

    Parámetros
    ----------
    valor_portafolio  : float → Valor actual del portafolio en MXN
    volatilidad_diaria: float → Desviación estándar diaria (ej. 0.02 = 2%)
    confianza         : float → Nivel de confianza (0.95 o 0.99)
    horizonte_dias    : int   → Horizonte de pérdida

    Retorna dict con VaR absoluto y relativo.
    """
    if valor_portafolio < 0:
        raise ValueError("valor_portafolio debe ser >= 0")
    if not (0 < confianza < 1):
        raise ValueError("confianza debe estar entre 0 y 1 (ej. 0.95)")
    if volatilidad_diaria < 0:
        raise ValueError("volatilidad_diaria debe ser >= 0")

    z = norm.ppf(confianza)
    var_relativo = volatilidad_diaria * np.sqrt(horizonte_dias) * z
    var_absoluto = valor_portafolio * var_relativo
    nivel = round(confianza * 100)   # round() evita int(0.99*100)=98 por float precision

    return {
        "Valor portafolio (MXN)": round(valor_portafolio, 2),
        "Volatilidad diaria (%)": round(volatilidad_diaria * 100, 2),
        "Confianza (%)": round(confianza * 100, 1),
        "Horizonte (días)": horizonte_dias,
        f"VaR {nivel}% (MXN)": round(var_absoluto, 2),
        f"VaR {nivel}% (%)": round(var_relativo * 100, 2),
        "Interpretación": (
            f"Con {nivel}% de confianza, la pérdida máxima esperada "
            f"en {horizonte_dias} día(s) es MXN {round(var_absoluto, 2):,.2f}"
        ),
    }
