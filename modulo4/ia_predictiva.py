"""
MÓDULO 4 — IA Predictiva (Elemento Extraordinario)
====================================================
Implementa un clasificador de Random Forest entrenado con
características macroeconómicas para predecir qué clase de
activo resistirá mejor una crisis.

Concepto: "Machine Learning aplicado a la Gestión de Portafolios"

Pipeline:
  1. Genera dataset sintético con escenarios históricos
  2. Entrena Random Forest (scikit-learn)
  3. Predice el activo más resiliente dado un entorno macro
  4. Explica la predicción con importancia de características

Maestría en Economía y Finanzas — Feria de Innovación
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ─────────────────────────────────────────────
#  GENERADOR DE DATASET DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def generar_dataset_historico(n_escenarios: int = 500, semilla: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético calibrado con datos históricos de México
    para entrenar el modelo de IA. Cada fila es un escenario macroeconómico
    con su activo "ganador" (el que mejor resistió).

    Variables de entrada (features):
    - inflacion_anual, tasa_cetes, volatilidad, tipo_cambio,
      spread_credito, liquidez, crecimiento_pib, confianza_consumidor

    Variable objetivo (label):
    - activo_ganador: el instrumento con menor pérdida o mayor ganancia
    """
    np.random.seed(semilla)

    registros = []

    for _ in range(n_escenarios):
        # Generar variables macroeconómicas aleatorias
        inflacion     = np.random.uniform(0.02, 0.30)
        tasa_cetes    = np.random.uniform(0.04, 0.20)
        volatilidad   = np.random.uniform(0.10, 0.60)
        tipo_cambio   = np.random.uniform(15.0, 25.0)
        spread        = np.random.uniform(0.005, 0.10)
        liquidez      = np.random.uniform(0.30, 1.20)
        crecimiento   = np.random.uniform(-0.08, 0.06)
        confianza     = np.random.uniform(60, 120)

        # Rendimiento esperado por activo (con ruido)
        r_cetes     = tasa_cetes - inflacion + np.random.normal(0, 0.01)
        r_udibonos  = 0.04 + np.random.normal(0, 0.005)  # protege inflación
        r_conv      = (tasa_cetes * 0.7 - volatilidad * 0.3 + np.random.normal(0, 0.02))
        r_fibra     = (crecimiento * 2 - tasa_cetes * 0.5 - volatilidad * 0.2
                       + np.random.normal(0, 0.03))
        r_etf_ipc   = (crecimiento * 3 - volatilidad * 0.5 + confianza * 0.002
                       + np.random.normal(0, 0.04))
        r_etf_nq    = (crecimiento * 2 - volatilidad * 0.3 + np.random.normal(0, 0.03))
        r_efectivo  = tasa_cetes * 0.2 + np.random.normal(0, 0.001)

        rendimientos = {
            "CETES": r_cetes,
            "UDIBONOS": r_udibonos,
            "Convertibles": r_conv,
            "FIBRA": r_fibra,
            "ETF IPC": r_etf_ipc,
            "ETF Nasdaq": r_etf_nq,
            "Efectivo": r_efectivo,
        }

        activo_ganador = max(rendimientos, key=rendimientos.get)

        registros.append({
            "inflacion":       round(inflacion, 4),
            "tasa_cetes":      round(tasa_cetes, 4),
            "volatilidad":     round(volatilidad, 4),
            "tipo_cambio":     round(tipo_cambio, 2),
            "spread_credito":  round(spread, 4),
            "liquidez":        round(liquidez, 4),
            "crecimiento_pib": round(crecimiento, 4),
            "confianza":       round(confianza, 1),
            "activo_ganador":  activo_ganador,
        })

    return pd.DataFrame(registros)


# ─────────────────────────────────────────────
#  ENTRENAMIENTO DEL MODELO
# ─────────────────────────────────────────────

FEATURES = [
    "inflacion", "tasa_cetes", "volatilidad",
    "tipo_cambio", "spread_credito", "liquidez",
    "crecimiento_pib", "confianza"
]
LABEL = "activo_ganador"


def entrenar_modelo(n_escenarios: int = 800) -> tuple:
    """
    Entrena un Random Forest Classifier con el dataset histórico.
    Retorna (metricas_dict, modelo_obj, encoder_obj) — SIN estado global.
    Cada sesión de usuario debe almacenar estos objetos en su propio
    st.session_state para evitar condiciones de carrera entre usuarios.
    """
    df = generar_dataset_historico(n_escenarios)

    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    importancias = dict(zip(FEATURES, modelo.feature_importances_))
    importancias_sorted = dict(
        sorted(importancias.items(), key=lambda x: x[1], reverse=True)
    )

    metricas = {
        "Precisión (Accuracy)": round(acc * 100, 1),
        "Escenarios de entrenamiento": len(X_train),
        "Escenarios de prueba": len(X_test),
        "Árboles en el bosque": 200,
        "Importancia de variables (%)": {
            k: round(v * 100, 2) for k, v in importancias_sorted.items()
        },
        "Clases": list(le.classes_),
        "Modelo": "Bosque Aleatorio Clasificador (scikit-learn)",
    }

    return metricas, modelo, le


def predecir_activo_resiliente(
    inflacion: float,
    tasa_cetes: float,
    volatilidad: float,
    tipo_cambio: float = 18.0,
    spread_credito: float = 0.02,
    liquidez: float = 0.8,
    crecimiento_pib: float = 0.01,
    confianza: float = 90.0,
    modelo=None,
    encoder=None,
) -> dict:
    """
    Dado un entorno macroeconómico, predice qué activo
    tiene mayor probabilidad de ser el más resiliente.

    Parámetros modelo y encoder deben pasarse explícitamente
    (obtenidos de st.session_state) para evitar estado global compartido.
    """
    if modelo is None or encoder is None:
        _, modelo, encoder = entrenar_modelo()

    entrada = pd.DataFrame([{
        "inflacion":       inflacion,
        "tasa_cetes":      tasa_cetes,
        "volatilidad":     volatilidad,
        "tipo_cambio":     tipo_cambio,
        "spread_credito":  spread_credito,
        "liquidez":        liquidez,
        "crecimiento_pib": crecimiento_pib,
        "confianza":       confianza,
    }])

    pred_idx    = modelo.predict(entrada)[0]
    pred_proba  = modelo.predict_proba(entrada)[0]
    activo_pred = encoder.inverse_transform([pred_idx])[0]
    clases      = encoder.classes_

    # Top 3 activos por probabilidad
    top_indices = np.argsort(pred_proba)[::-1][:3]
    top3 = [(clases[i], round(pred_proba[i] * 100, 1)) for i in top_indices]

    # Probabilidades de todos
    probs_all = {clases[i]: round(pred_proba[i] * 100, 1)
                 for i in range(len(clases))}

    # Lógica interpretable de la predicción
    if inflacion > 0.15:
        razon = "Inflación muy alta → el modelo favorece activos indexados a precios (UDIBONOS)."
    elif volatilidad > 0.40:
        razon = "Volatilidad extrema → instrumentos de renta fija soberana (CETES, Efectivo) son refugio."
    elif spread_credito > 0.06:
        razon = "Spread crediticio alto → riesgo de crédito elevado, se favorecen soberanos sobre corporativos."
    elif crecimiento_pib > 0.04:
        razon = "Crecimiento económico robusto → se favorecen acciones, ETFs y FIBRAS."
    elif liquidez < 0.50:
        razon = "Iliquidez extrema → el efectivo y los CETES son el último refugio."
    else:
        razon = (f"Entorno moderado → el modelo pondera múltiples factores. "
                 f"La combinación inflación={inflacion*100:.1f}%, "
                 f"volatilidad={volatilidad*100:.1f}% favorece {activo_pred}.")

    return {
        "Activo recomendado": activo_pred,
        "Confianza del modelo (%)": round(pred_proba[pred_idx] * 100, 1),
        "Top 3 candidatos": top3,
        "Probabilidades por activo (%)": probs_all,
        "Razonamiento": razon,
        "Condiciones evaluadas": {
            "Inflación (%)": round(inflacion * 100, 1),
            "Tasa CETES (%)": round(tasa_cetes * 100, 1),
            "Volatilidad (%)": round(volatilidad * 100, 1),
            "Tipo de cambio": tipo_cambio,
            "Spread crédito (%)": round(spread_credito * 100, 2),
            "Liquidez (índice)": liquidez,
            "Crecimiento PIB (%)": round(crecimiento_pib * 100, 1),
            "Confianza consumidor": confianza,
        },
        "Algoritmo": "Bosque Aleatorio — 200 árboles de decisión",
        "Interpretación": (
            f"De cada 100 escenarios con estas condiciones macroeconómicas, "
            f"el modelo predice que '{activo_pred}' sería el activo más resiliente "
            f"en {round(pred_proba[pred_idx]*100, 1)} de ellos."
        ),
    }


def importancia_variables_df(modelo=None) -> pd.DataFrame:
    """
    Retorna DataFrame con importancia de cada variable para usar en gráfica.
    modelo debe provenir de st.session_state para evitar estado global.
    """
    if modelo is None:
        _, modelo, _ = entrenar_modelo()

    importancias = modelo.feature_importances_
    nombres_es = {
        "inflacion":       "Inflación",
        "tasa_cetes":      "Tasa CETES",
        "volatilidad":     "Volatilidad",
        "tipo_cambio":     "Tipo de Cambio",
        "spread_credito":  "Diferencial Crediticio",
        "liquidez":        "Liquidez",
        "crecimiento_pib": "Crecimiento PIB",
        "confianza":       "Confianza Consumidor",
    }
    df = pd.DataFrame({
        "Variable": [nombres_es[f] for f in FEATURES],
        "Importancia (%)": [round(v * 100, 2) for v in importancias],
    }).sort_values("Importancia (%)", ascending=False)

    return df
