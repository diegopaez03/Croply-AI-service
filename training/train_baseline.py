#!/usr/bin/env python3
"""
train_baseline.py
-----------------
Entrena, evalúa y serializa el modelo clasificador multi-clase de acciones agronómicas
para Croply utilizando LightGBM sobre el dataset híbrido calibrado para Cuyo (Mendoza).

Ajustes y blindaje en producción (v1.2.0):
1. Soporte para fallo en sensor de Humedad de Suelo (humedad_suelo_pct y deficit_hidrico con NaNs).
2. Blindaje de get_dynamic_sensor_origin():
   - Descarta features que tengan NaN en la fila evaluada.
   - Mapea variables derivadas a sensores físicos reales de la UI (PAN-AF-03).
3. Inferencia autónoma de severidad (infer_severity()) sin recurrir al target del CSV.
4. Generación de bandera requiere_intervencion_llm según HU-NA-03 para evitar
   llamadas redundantes a Gemma 4 en Ollama cuando la parcela vegeta en estado óptimo.
5. Test unitario con dos casos representativos de sensores faltantes:
   - Caso 1: Suelo seco con fallo en pH y Radiación -> predice REGAR_* con sensor humedad_suelo_pct.
   - Caso 2: Fallo en sensor de humedad de suelo + viento alto (28 km/h) y cura pendiente -> predice POSTERGAR_CURACION_CLIMA con sensor viento_kmh.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "raw_datasets", "croply_training_dataset_cuyo.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "app", "models")
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "croply_classifier_v1.joblib")
PAYLOAD_OUT_PATH = os.path.join(BASE_DIR, "..", "data", "raw_datasets", "sample_payload_recommender.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# 20 Features de Entrada
FEATURE_COLS = [
    "cultivo_enc",
    "dias_desde_siembra",
    "etapa_fenologica_idx",
    "humedad_suelo_pct",
    "temp_ambiente_c",
    "humedad_ambiente_pct",
    "radiacion_solar_wm2",
    "ph_suelo",
    "lluvia_mm_24h",
    "viento_kmh",
    "wind_gust_kmh",
    "temp_min_pronostico",
    "temp_max_pronostico",
    "precip_pronostico_24h",
    "et0_diario",
    "deficit_hidrico",
    "delta_t",
    "dias_desde_ultimo_riego",
    "dias_desde_ultima_curacion",
    "hora_del_dia"
]

# Variables estáticas o no sensoriales a excluir al determinar sensor_origen_principal
STATIC_NON_SENSOR_COLS = {
    "cultivo_enc",
    "dias_desde_siembra",
    "etapa_fenologica_idx",
    "hora_del_dia",
    "dias_desde_ultimo_riego",
    "dias_desde_ultima_curacion"
}

# Mapeo de variables derivadas/ingenieriles al sensor físico correspondiente en UI (PAN-AF-03)
SENSOR_UI_MAPPING = {
    "deficit_hidrico": "humedad_suelo_pct",
    "delta_t": "temp_ambiente_c",
    "wind_gust_kmh": "viento_kmh",
    "precip_pronostico_24h": "lluvia_mm_24h"
}

def load_and_preprocess_data():
    logging.info(f"Cargando dataset desde {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    logging.info(f"Dataset cargado con {len(df)} filas y {len(df.columns)} columnas.")
    
    # Label Encoders
    le_cultivo = LabelEncoder()
    df["cultivo_enc"] = le_cultivo.fit_transform(df["cultivo"])
    
    le_target = LabelEncoder()
    df["target_enc"] = le_target.fit_transform(df["target_accion"])
    
    return df, le_cultivo, le_target

def get_dynamic_sensor_origin(feature_names, shap_values_for_class, sample_row):
    """
    Determina de forma dinámica cuál fue el sensor o variable física principal
    que disparó la predicción:
    1. Descarta variables estáticas de contexto.
    2. Descarta features que en sample_row sean NaN (sensores faltantes/sin señal).
    3. Mapea variables derivadas al sensor físico de la UI (PAN-AF-03).
    """
    best_feature = "humedad_suelo_pct"
    max_impact = -1.0
    
    for feat_name, shap_val in zip(feature_names, shap_values_for_class):
        if feat_name in STATIC_NON_SENSOR_COLS:
            continue
            
        # Validar si el sensor vino en NaN
        val = sample_row.get(feat_name, None)
        if val is None or pd.isna(val):
            continue
            
        if abs(shap_val) > max_impact:
            max_impact = abs(shap_val)
            best_feature = feat_name
            
    # Mapeo al sensor físico real de la interfaz
    return SENSOR_UI_MAPPING.get(best_feature, best_feature)

def apply_feasibility_mask(proba_vector, target_names, sample_row, cultivo_str=None, etapa_idx=None):
    """
    Guardarraíl post-predicción para inferencia y evaluación (HU-NA-03):
    Anula físicamente las probabilidades de acciones agronómicas imposibles según los sensores presentes:
    1. Si ph_suelo es NaN: P(CORREGIR_PH_SUELO) = 0.
    2. Si humedad_suelo_pct es NaN:
       - P(REGAR_URGENTE) = 0
       - P(REGAR_PROGRAMADO) = 0
       - P(SUSPENDER_RIEGO) = 0 a menos que haya lluvia pronosticada copiosa (>= 20mm) o sea etapa pre-cosecha (etapa 3 de ajo/cebolla).
    3. Normaliza las probabilidades restantes para que sumen 1.0 (o asigna 1.0 a ESTADO_OPTIMO si todo se anula).
    """
    proba = np.array(proba_vector, dtype=float).copy()
    
    val_ph = sample_row.get("ph_suelo", np.nan)
    has_ph = (val_ph is not None and not pd.isna(val_ph))
    
    val_soil = sample_row.get("humedad_suelo_pct", np.nan)
    has_soil = (val_soil is not None and not pd.isna(val_soil))
    
    precip_24h = sample_row.get("precip_pronostico_24h", 0.0)
    if precip_24h is None or pd.isna(precip_24h):
        precip_24h = 0.0
        
    for i, action in enumerate(target_names):
        if action == "CORREGIR_PH_SUELO" and not has_ph:
            proba[i] = 0.0
        elif not has_soil:
            if action in ["REGAR_URGENTE", "REGAR_PROGRAMADO"]:
                proba[i] = 0.0
            elif action == "SUSPENDER_RIEGO":
                es_precosecha_bulbos = (cultivo_str in ["ajo", "cebolla"] and etapa_idx == 3)
                if not (precip_24h >= 20.0 or es_precosecha_bulbos):
                    proba[i] = 0.0
                    
    total = np.sum(proba)
    if total > 0:
        proba = proba / total
    else:
        if "ESTADO_OPTIMO_SIN_ACCION" in target_names:
            proba[target_names.index("ESTADO_OPTIMO_SIN_ACCION")] = 1.0
        else:
            proba = np.ones_like(proba) / len(proba)
            
    return proba

def infer_severity(pred_action, sample_row):
    """
    Infiere dinámicamente la severidad ('Alta', 'Media', 'Baja') en tiempo de ejecución
    a partir de la acción predicha y las lecturas de telemetría, sin depender del target del dataset.
    """
    if pred_action == "REGAR_URGENTE":
        return "Alta"
    elif pred_action == "POSTERGAR_CURACION_CLIMA":
        viento = sample_row.get("viento_kmh", 0.0)
        temp = sample_row.get("temp_ambiente_c", 20.0)
        if (viento is not None and not pd.isna(viento) and viento >= 35.0) or (temp is not None and not pd.isna(temp) and temp >= 35.0):
            return "Alta"
        return "Media"
    elif pred_action == "SUSPENDER_RIEGO":
        precip = sample_row.get("precip_pronostico_24h", 0.0)
        theta = sample_row.get("humedad_suelo_pct", 0.0)
        if (precip is not None and not pd.isna(precip) and precip >= 30.0) or (theta is not None and not pd.isna(theta) and theta >= 33.0):
            return "Alta"
        return "Media"
    elif pred_action == "CORREGIR_PH_SUELO":
        return "Media"
    elif pred_action == "REGAR_PROGRAMADO":
        return "Media"
    elif pred_action == "VENTANA_OPTIMA_CURAR":
        return "Baja"
    elif pred_action == "ESTADO_OPTIMO_SIN_ACCION":
        return "Baja"
    return "Media"

def get_deterministic_fallback(pred_action, inferred_severity, sensor_origin, sample_row, cultivo, etapa):
    """
    Genera la respuesta determinista de respaldo (HU-NA-03) si Gemma 4 en Ollama
    demora más de 30 segundos o experimenta un error de servicio [ERR-01].
    Garantiza disponibilidad ininterrumpida de recomendaciones agronómicas.
    """
    cultivo_title = str(cultivo).capitalize()
    
    theta = sample_row.get("humedad_suelo_pct", np.nan)
    theta_str = f"{theta:.1f}%" if (theta is not None and not pd.isna(theta)) else "sin telemetría directa"
    
    viento = sample_row.get("viento_kmh", np.nan)
    viento_str = f"{viento:.1f} km/h" if (viento is not None and not pd.isna(viento)) else "velocidad no informada"
    
    ph = sample_row.get("ph_suelo", np.nan)
    ph_str = f"{ph:.2f}" if (ph is not None and not pd.isna(ph)) else "no disponible"
    
    if pred_action == "REGAR_URGENTE":
        return {
            "titulo_corto": "Riego urgente recomendado",
            "texto_breve": f"Humedad del suelo ({theta_str}) bajo umbral crítico en {etapa}.",
            "texto_explicativo_extendido": (
                f"El cultivo de {cultivo_title} se encuentra en etapa de {etapa} y presenta un déficit hídrico severo detectado por el sensor {sensor_origin} ({theta_str}). "
                "Se aconseja activar inmediatamente el turno de riego para prevenir estrés hídrico irreversible y salvaguardar el rendimiento proyectado."
            )
        }
    elif pred_action == "REGAR_PROGRAMADO":
        return {
            "titulo_corto": "Riego preventivo programado",
            "texto_breve": f"Humedad ({theta_str}) próxima a agotar el agua fácilmente utilizable.",
            "texto_explicativo_extendido": (
                f"La reserva hídrica de la parcela para {cultivo_title} está descendiendo gradualmente hacia el límite de reposición recomendado por INTA. "
                "Programe el próximo turno de riego en las próximas 24 a 48 horas para mantener la zona radicular en capacidad de campo."
            )
        }
    elif pred_action == "SUSPENDER_RIEGO":
        precip = sample_row.get("precip_pronostico_24h", 0.0)
        precip_val = 0.0 if (precip is None or pd.isna(precip)) else float(precip)
        return {
            "titulo_corto": "Suspender riego programado",
            "texto_breve": f"Exceso de humedad ({theta_str}) o lluvias pronosticadas ({precip_val:.1f} mm).",
            "texto_explicativo_extendido": (
                f"Se recomienda cancelar o postergar el riego en el lote de {cultivo_title}. "
                f"La telemetría de {sensor_origin} indica saturación o aportes pluviales inminentes suficientes para evitar asfixia radicular y enfermedades fúngicas."
            )
        }
    elif pred_action == "VENTANA_OPTIMA_CURAR":
        return {
            "titulo_corto": "Ventana óptima para pulverizar",
            "texto_breve": f"Condiciones climáticas ideales: viento a {viento_str} y delta T apto.",
            "texto_explicativo_extendido": (
                f"Las condiciones meteorológicas actuales para {cultivo_title} son favorables para realizar aplicaciones fitosanitarias programadas. "
                f"El viento ({viento_str}) y la psicrometría aseguran una correcta cobertura foliar con deriva y evaporación mínimas."
            )
        }
    elif pred_action == "POSTERGAR_CURACION_CLIMA":
        return {
            "titulo_corto": "Postergar aplicación sanitaria",
            "texto_breve": f"Riesgo meteorológico adverso por variable {sensor_origin}.",
            "texto_explicativo_extendido": (
                f"Se desaconseja efectuar pulverizaciones sobre {cultivo_title} en este momento. "
                f"La lectura del sensor {sensor_origin} excede los límites seguros de aplicación, lo que provocaría pérdidas significativas por deriva o evaporación prematura."
            )
        }
    elif pred_action == "CORREGIR_PH_SUELO":
        return {
            "titulo_corto": "Corrección de pH requerida",
            "texto_breve": f"pH del suelo ({ph_str}) desviado del rango óptimo para {cultivo_title}.",
            "texto_explicativo_extendido": (
                f"La lectura del sensor {sensor_origin} reporta un pH de {ph_str}, fuera del intervalo ideal para el cultivo de {cultivo_title}. "
                "Se sugiere evaluar la aplicación de enmiendas edáficas o acidificantes en el agua de riego para optimizar la biodisponibilidad de nutrientes."
            )
        }
    else: # ESTADO_OPTIMO_SIN_ACCION
        return {
            "titulo_corto": "Cultivo en condición óptima",
            "texto_breve": "Todos los parámetros edafoclimáticos dentro de los rangos meta.",
            "texto_explicativo_extendido": (
                f"El lote de {cultivo_title} en etapa de {etapa} no presenta necesidades de intervención agronómica urgente. "
                "Continúe con el esquema habitual de monitoreo y telemetría de Croply."
            )
        }

def test_missing_sensors_inference(clf, le_cultivo, target_names, explainer):
    """
    Prueba unitaria para verificar que el modelo, el guardarraíl de viabilidad física y SHAP
    manejan apropiadamente muestras en producción donde faltan sensores:
    - Caso 1: Suelo seco con sensores de pH y Radiación desconectados (NaN).
    - Caso 2: Sensor de Humedad de Suelo desconectado (humedad=NaN, deficit=NaN),
              pero hay viento alto (28 km/h) y cura pendiente (16 días).
    - Caso 3: Guardarraíl de viabilidad física (apply_feasibility_mask) anula estrictamente
              probabilidades residuales sobre sensores en NaN.
    """
    logging.info("--- INICIANDO TEST UNITARIO: INFERENCIA CON SENSORES FALTANTES (NaN) ---")
    cultivo_idx = int(le_cultivo.transform(["tomate"])[0])
    
    # -------------------------------------------------------------
    # CASO 1: pH y Radiación en NaN, Suelo seco
    # -------------------------------------------------------------
    test_sample_1 = pd.DataFrame([{
        "cultivo_enc": cultivo_idx,
        "dias_desde_siembra": 50,
        "etapa_fenologica_idx": 2,          # Floración
        "humedad_suelo_pct": 16.5,          # Suelo seco
        "temp_ambiente_c": 26.0,
        "humedad_ambiente_pct": 45.0,
        "radiacion_solar_wm2": np.nan,      # SENSOR FALTANTE
        "ph_suelo": np.nan,                 # SENSOR FALTANTE
        "lluvia_mm_24h": 0.0,
        "viento_kmh": 8.0,
        "wind_gust_kmh": 11.0,
        "temp_min_pronostico": 15.0,
        "temp_max_pronostico": 29.0,
        "precip_pronostico_24h": 0.0,
        "et0_diario": 5.8,
        "deficit_hidrico": 8.7,             # 25.2 - 16.5 = +8.7%
        "delta_t": 5.2,
        "dias_desde_ultimo_riego": 6,
        "dias_desde_ultima_curacion": 5,
        "hora_del_dia": 10
    }])[FEATURE_COLS]
    
    proba_raw_1 = clf.predict_proba(test_sample_1)[0]
    proba_1 = apply_feasibility_mask(proba_raw_1, target_names, test_sample_1.iloc[0], "tomate", 2)
    pred_idx_1 = int(np.argmax(proba_1))
    pred_action_1 = target_names[pred_idx_1]
    
    shap_vals_1 = explainer.shap_values(test_sample_1)
    if isinstance(shap_vals_1, list):
        class_shap_1 = shap_vals_1[pred_idx_1][0]
    elif len(shap_vals_1.shape) == 3:
        class_shap_1 = shap_vals_1[0, :, pred_idx_1]
    else:
        class_shap_1 = shap_vals_1[0]
        
    sensor_origen_1 = get_dynamic_sensor_origin(FEATURE_COLS, class_shap_1, test_sample_1.iloc[0])
    
    logging.info(f"Resultado Caso 1 (pH y Radiación NaN):")
    logging.info(f"  Acción predicha: {pred_action_1} (Confianza: {proba_1[pred_idx_1]:.2%})")
    logging.info(f"  Sensor de origen: {sensor_origen_1}")
    assert not np.isnan(proba_1).any(), "Error: Probabilidades contienen NaN en Caso 1"
    assert np.isclose(np.sum(proba_1), 1.0), "Error: Probabilidades en Caso 1 no suman 1.0"
    assert proba_1[target_names.index("CORREGIR_PH_SUELO")] == 0.0, "Error: Guardarraíl no anuló CORREGIR_PH_SUELO sin sensor pH"
    assert pred_action_1 in ["REGAR_URGENTE", "REGAR_PROGRAMADO"], f"Inesperada acción Caso 1: {pred_action_1}"
    assert sensor_origen_1 == "humedad_suelo_pct", f"Inesperado sensor origen: {sensor_origen_1}"

    # -------------------------------------------------------------
    # CASO 2: Humedad de Suelo y Déficit en NaN, Viento Alto + Cura Pendiente
    # -------------------------------------------------------------
    test_sample_2 = pd.DataFrame([{
        "cultivo_enc": cultivo_idx,
        "dias_desde_siembra": 60,
        "etapa_fenologica_idx": 2,          # Floración
        "humedad_suelo_pct": np.nan,        # SENSOR DE SUELO FALTANTE
        "temp_ambiente_c": 24.5,
        "humedad_ambiente_pct": 42.0,
        "radiacion_solar_wm2": 650.0,
        "ph_suelo": 6.8,
        "lluvia_mm_24h": 0.0,
        "viento_kmh": 28.0,                 # VIENTO FUERTE (supera límite de 15 km/h)
        "wind_gust_kmh": 36.0,
        "temp_min_pronostico": 14.0,
        "temp_max_pronostico": 28.0,
        "precip_pronostico_24h": 0.0,
        "et0_diario": 6.2,
        "deficit_hidrico": np.nan,          # DÉFICIT FALTANTE CONSECUENTE
        "delta_t": 5.8,
        "dias_desde_ultimo_riego": 4,
        "dias_desde_ultima_curacion": 16,   # CURACIÓN PENDIENTE (>= 14 días)
        "hora_del_dia": 11
    }])[FEATURE_COLS]
    
    proba_raw_2 = clf.predict_proba(test_sample_2)[0]
    proba_2 = apply_feasibility_mask(proba_raw_2, target_names, test_sample_2.iloc[0], "tomate", 2)
    pred_idx_2 = int(np.argmax(proba_2))
    pred_action_2 = target_names[pred_idx_2]
    
    shap_vals_2 = explainer.shap_values(test_sample_2)
    if isinstance(shap_vals_2, list):
        class_shap_2 = shap_vals_2[pred_idx_2][0]
    elif len(shap_vals_2.shape) == 3:
        class_shap_2 = shap_vals_2[0, :, pred_idx_2]
    else:
        class_shap_2 = shap_vals_2[0]
        
    sensor_origen_2 = get_dynamic_sensor_origin(FEATURE_COLS, class_shap_2, test_sample_2.iloc[0])
    
    logging.info(f"Resultado Caso 2 (Humedad Suelo NaN + Viento Alto + Cura Pendiente):")
    logging.info(f"  Acción predicha: {pred_action_2} (Confianza: {proba_2[pred_idx_2]:.2%})")
    logging.info(f"  Sensor de origen: {sensor_origen_2}")
    assert not np.isnan(proba_2).any(), "Error: Probabilidades contienen NaN en Caso 2"
    assert np.isclose(np.sum(proba_2), 1.0), "Error: Probabilidades en Caso 2 no suman 1.0"
    assert proba_2[target_names.index("REGAR_URGENTE")] == 0.0, "Error: Guardarraíl no anuló REGAR_URGENTE sin sensor suelo"
    assert proba_2[target_names.index("REGAR_PROGRAMADO")] == 0.0, "Error: Guardarraíl no anuló REGAR_PROGRAMADO sin sensor suelo"
    assert pred_action_2 == "POSTERGAR_CURACION_CLIMA", f"Inesperada acción Caso 2: {pred_action_2}"
    assert sensor_origen_2 == "viento_kmh", f"Inesperado sensor origen Caso 2: {sensor_origen_2}"

    # -------------------------------------------------------------
    # CASO 3: Validación Directa del Guardarraíl con Vector Sintético
    # -------------------------------------------------------------
    mock_proba = np.ones(len(target_names)) / len(target_names) # 1/7 para cada clase
    mock_row_no_sensors = {
        "humedad_suelo_pct": np.nan,
        "ph_suelo": np.nan,
        "precip_pronostico_24h": 0.0
    }
    masked_proba = apply_feasibility_mask(mock_proba, target_names, mock_row_no_sensors, "tomate", 2)
    assert np.isclose(np.sum(masked_proba), 1.0), "Error: Mock proba en Caso 3 no normaliza a 1.0"
    assert masked_proba[target_names.index("CORREGIR_PH_SUELO")] == 0.0, "Fallo: pH NaN no anuló CORREGIR_PH_SUELO"
    assert masked_proba[target_names.index("REGAR_URGENTE")] == 0.0, "Fallo: Suelo NaN no anuló REGAR_URGENTE"
    assert masked_proba[target_names.index("REGAR_PROGRAMADO")] == 0.0, "Fallo: Suelo NaN no anuló REGAR_PROGRAMADO"
    assert masked_proba[target_names.index("SUSPENDER_RIEGO")] == 0.0, "Fallo: Suelo NaN sin lluvia no anuló SUSPENDER_RIEGO"
    logging.info("Validación exitosa Caso 3: Guardarraíl apply_feasibility_mask verificado unitariamente.")

    logging.info("--- TEST UNITARIO COMPLETADO CON ÉXITO: TODOS LOS CASOS VALIDADOS ---\n")

def train_and_evaluate():
    df, le_cultivo, le_target = load_and_preprocess_data()
    
    X = df[FEATURE_COLS]
    y = df["target_enc"]
    
    # Split Estratificado 80 / 20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    logging.info(f"Datos de entrenamiento: {len(X_train)} | Datos de test: {len(X_test)}")
    
    # Hiperparámetros calibrados de LightGBM
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le_target.classes_),
        n_estimators=250,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    
    logging.info("Entrenando modelo LightGBM...")
    clf.fit(X_train, y_train)
    logging.info("Entrenamiento finalizado exitosamente.")
    
    target_names = [str(cls) for cls in le_target.classes_]
    
    # Evaluación en Test con Guardarraíl de Viabilidad Física
    y_proba_raw = clf.predict_proba(X_test)
    y_proba = np.zeros_like(y_proba_raw)
    for i in range(len(X_test)):
        sample_row_eval = X_test.iloc[i]
        c_idx = int(sample_row_eval["cultivo_enc"])
        cultivo_str = le_cultivo.inverse_transform([c_idx])[0]
        etapa_idx = int(sample_row_eval["etapa_fenologica_idx"])
        y_proba[i] = apply_feasibility_mask(y_proba_raw[i], target_names, sample_row_eval, cultivo_str, etapa_idx)
    y_pred = np.argmax(y_proba, axis=1)
    
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    
    print("\n" + "="*70)
    print(f"MÉTRICAS DE RENDIMIENTO GLOBAL DEL MODELO CROPLY (v1.3.0)")
    print(f"Macro F1-Score:    {macro_f1:.4f} (Objetivo de producción >= 0.85)")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print("="*70)
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\n--- REPORTE DETALLADO DE CLASIFICACIÓN POR ACCIÓN ---")
    print(report)
    
    print("\n--- MATRIZ DE CONFUSIÓN (Test Set) ---")
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(df_cm.to_string())
    
    # Feature Importance (Gain y Split)
    importance_gain = clf.booster_.feature_importance(importance_type="gain")
    df_importance = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Gain_Importance": importance_gain
    }).sort_values(by="Gain_Importance", ascending=False)
    
    print("\n--- TOP 10 FEATURES MÁS DETERMINANTES (GAIN IMPORTANCE) ---")
    print(df_importance.head(10).to_string(index=False))
    
    # Serialización del Modelo y Encoders
    artifacts = {
        "model": clf,
        "feature_cols": FEATURE_COLS,
        "le_cultivo": le_cultivo,
        "le_target": le_target,
        "target_names": target_names,
        "edaphic_constants": {
            "THETA_CC": 30.0,
            "THETA_PMP": 14.0,
            "AU_TOTAL": 16.0
        },
        "metrics": {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        }
    }
    joblib.dump(artifacts, MODEL_OUT_PATH)
    logging.info(f"Modelo y artefactos guardados en: {os.path.abspath(MODEL_OUT_PATH)}")
    
    # Explicabilidad SHAP y Generación del Payload de Ejemplo para recommender.py
    logging.info("Calculando explicabilidad local con SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    
    # Ejecutar Test Unitario de Sensores Faltantes (Casos 1, 2 y 3)
    test_missing_sensors_inference(clf, le_cultivo, target_names, explainer)
    
    # Seleccionamos un caso de prueba del test set
    test_idx = 0
    sample_x = X_test.iloc[[test_idx]]
    sample_raw = df.iloc[X_test.index[test_idx]]
    
    # Predicción y aplicación estricta del guardarraíl de viabilidad
    proba_sample_raw = clf.predict_proba(sample_x)[0]
    cultivo_sample_str = str(sample_raw["cultivo"])
    etapa_sample_idx = int(sample_raw["etapa_fenologica_idx"])
    proba_sample = apply_feasibility_mask(proba_sample_raw, target_names, sample_x.iloc[0], cultivo_sample_str, etapa_sample_idx)
    pred_class_idx = int(np.argmax(proba_sample))
    pred_action = target_names[pred_class_idx]
    confidence = float(proba_sample[pred_class_idx])
    
    # SHAP values para este sample
    shap_vals_sample = explainer.shap_values(sample_x)
    if isinstance(shap_vals_sample, list):
        class_shap = shap_vals_sample[pred_class_idx][0]
    elif len(shap_vals_sample.shape) == 3:
        class_shap = shap_vals_sample[0, :, pred_class_idx]
    else:
        class_shap = shap_vals_sample[0]
        
    # Top 3 features explicativas globales (sin NaNs y sin redundancia semántica)
    SEMANTIC_GROUPS = {
        "humedad_suelo_pct": "suelo_humedad",
        "deficit_hidrico": "suelo_humedad",
        "viento_kmh": "viento",
        "wind_gust_kmh": "viento",
        "temp_ambiente_c": "temperatura",
        "delta_t": "temperatura"
    }

    sorted_indices = np.argsort(np.abs(class_shap))[::-1]
    top_features = []
    seen_groups = set()
    
    for idx_feat in sorted_indices:
        feat_name = FEATURE_COLS[idx_feat]
        val = sample_x.iloc[0][feat_name]
        
        # 1. Ignorar features que sean NaN (sensores desconectados)
        if pd.isna(val):
            continue
            
        # 2. Ignorar si ya incluimos una variable del mismo grupo semántico
        group = SEMANTIC_GROUPS.get(feat_name, feat_name)
        if group in seen_groups:
            continue
            
        seen_groups.add(group)
        top_features.append({
            "feature": feat_name,
            "valor": float(val),
            "shap_attribution": round(float(class_shap[idx_feat]), 4)
        })
        
        if len(top_features) == 3:
            break
        
    # DERIVACIÓN DINÁMICA DE SENSOR_ORIGEN_PRINCIPAL (BLINDADA CONTRA NaNs Y MAPEO UI)
    dynamic_sensor_origin = get_dynamic_sensor_origin(FEATURE_COLS, class_shap, sample_x.iloc[0])
    
    # INFERENCIA DINÁMICA DE SEVERIDAD (SIN LEAKAGE)
    inferred_severity = infer_severity(pred_action, sample_x.iloc[0])
    
    # EVALUACIÓN BOOLEANA DE ALERTA CLIMÁTICA Y CONDICIÓN DE LLAMADA A GEMMA 4 (HU-NA-03)
    alerta_raw = str(sample_raw["alerta_climatica"])
    has_real_alert = (alerta_raw != "SIN_ALERTA") and ("ALERTA" in alerta_raw)
    
    # Si la parcela está en estado óptimo y sin alertas, no se invoca innecesariamente al LLM
    requiere_intervencion_llm = bool(pred_action != "ESTADO_OPTIMO_SIN_ACCION" or has_real_alert)
    
    # RESPUESTA DETERMINISTA DE RESPALDO (FALLBACK EN CASO DE TIMEOUT >30s O ERROR EN OLLAMA)
    fallback_determinista = get_deterministic_fallback(
        pred_action, inferred_severity, dynamic_sensor_origin, sample_x.iloc[0],
        str(sample_raw["cultivo"]), str(sample_raw["etapa_fenologica"])
    )

    # Construir Payload JSON Estructurado para recommender.py (Gemma 4 en Ollama)
    recommender_payload = {
        "metadata": {
            "version_analyzer": "1.3.0-lgbm-loam-prod",
            "parcela_id": "mza-guaymallen-lote-04",
            "cultivo": str(sample_raw["cultivo"]),
            "etapa_fenologica": str(sample_raw["etapa_fenologica"]),
            "dias_desde_siembra": int(sample_raw["dias_desde_siembra"]),
            "timestamp": "2026-10-17T18:30:00-03:00"
        },
        "prediccion_ml": {
            "accion_recomendada": pred_action,
            "severidad": inferred_severity,
            "confianza": round(confidence, 4),
            "sensor_origen_principal": dynamic_sensor_origin,
            "requiere_intervencion_llm": requiere_intervencion_llm,
            "distribucion_probabilidad": {
                target_names[i]: round(float(proba_sample[i]), 4)
                for i in range(len(target_names))
            },
            "explicabilidad_top3_factores": top_features
        },
        "response_schema_esperado": {
            "tipo": "object",
            "campos_requeridos": [
                "titulo_corto",
                "texto_breve",
                "texto_explicativo_extendido"
            ],
            "especificaciones": {
                "titulo_corto": "Título imperativo/badge, máximo 40 caracteres (ej. 'Riego urgente recomendado').",
                "texto_breve": "Resumen directo con datos clave del sensor principal, máximo 120 caracteres para tarjetas UI o notificaciones push.",
                "texto_explicativo_extendido": "Explicación agronómica y empática de 2 a 3 oraciones contextualizada a la etapa fenológica actual."
            }
        },
        "fallback_determinista": fallback_determinista,
        "alertas_climaticas_activas": [
            alerta for alerta in alerta_raw.split("|") if alerta != "SIN_ALERTA"
        ],
        "riesgo_climatico_fenologico": {
            "nivel": "ALTO" if has_real_alert else "BAJO",
            "impacto": "Condición climática evaluada con sensibilidad de etapa " + str(sample_raw["etapa_fenologica"])
        },
        "telemetria_actual": {
            "humedad_suelo_pct": None if pd.isna(sample_raw["humedad_suelo_pct"]) else float(sample_raw["humedad_suelo_pct"]),
            "temp_ambiente_c": float(sample_raw["temp_ambiente_c"]),
            "humedad_ambiente_pct": float(sample_raw["humedad_ambiente_pct"]),
            "radiacion_solar_wm2": None if pd.isna(sample_raw["radiacion_solar_wm2"]) else float(sample_raw["radiacion_solar_wm2"]),
            "ph_suelo": None if pd.isna(sample_raw["ph_suelo"]) else float(sample_raw["ph_suelo"]),
            "viento_kmh": float(sample_raw["viento_kmh"]),
            "wind_gust_kmh": float(sample_raw["wind_gust_kmh"]),
            "delta_t_psicrometrico": float(sample_raw["delta_t"]),
            "deficit_hidrico_pct": None if pd.isna(sample_raw["deficit_hidrico"]) else float(sample_raw["deficit_hidrico"])
        },
        "umbrales_agronomicos_referencia_inta": {
            "suelo_tipo": "Franco",
            "capacidad_campo_pct": 30.0,
            "punto_marchitez_pct": 14.0,
            "agua_util_pct": 16.0,
            "rango_ph_optimo": [6.0, 6.8] if sample_raw["cultivo"] == "tomate" else [6.0, 7.0],
            "viento_maximo_pulverizacion_kmh": 15.0,
            "delta_t_optimo_pulverizacion": [2.0, 8.0]
        },
        "instrucciones_para_gemma4": {
            "regla_oro": "No inventar datos ni contradecir la accion_recomendada ni las alertas_climaticas_activas.",
            "enfoque": "Generar obligatoriamente un JSON que cumpla response_schema_esperado con titulo_corto, texto_breve y texto_explicativo_extendido, citando sensor_origen_principal.",
            "tiempo_maximo_redaccion_seg": 3,
            "estado_vacio_ui": "Si requiere_intervencion_llm es False, el frontend no invoca al LLM y muestra: 'No hay recomendaciones activas en este momento.'"
        }
    }
    
    with open(PAYLOAD_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(recommender_payload, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Payload de ejemplo para recommender.py exportado a: {os.path.abspath(PAYLOAD_OUT_PATH)}")
    print("\n--- EJEMPLO DE PAYLOAD EXPORTADO PARA RECOMMENDER.PY (GEMMA 4) ---")
    print(json.dumps(recommender_payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    train_and_evaluate()
