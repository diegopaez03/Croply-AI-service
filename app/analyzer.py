"""
Croply - Servicio Analítico de Inteligencia Artificial (analyzer.py)
===================================================================
Módulo analítico determinista y predictivo basado en LightGBM y SHAP.
Carga el modelo serializado croply_classifier_v1.joblib como singleton en memoria
para garantizar latencias de inferencia inferiores a 5 ms.

Responsabilidades:
1. Preprocesamiento e ingeniería de features en tiempo real.
2. Detección determinista de alertas climáticas (HU-NA-01).
3. Inferencia de acción agronómica recomendada con guardarraíl de viabilidad física (apply_feasibility_mask).
4. Explicabilidad local con SHAP TreeExplainer (filtrado de NaNs, deduplicación semántica y prioridad SHAP > 0).
5. Generación de fallback determinista basado en reglas agronómicas de INTA / Cuyo.
"""

import os
import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constantes edafológicas para suelo Franco de Cuyo (Mendoza)
THETA_CC = 30.0   # Capacidad de Campo (% volumétrico)
THETA_PMP = 14.0  # Punto de Marchitez Permanente (% volumétrico)
AU_TOTAL = 16.0   # Agua Útil Total (%)

# Configuración fenológica y agronómica por cultivo (INTA Mendoza / FAO-56)
CROPS_CONFIG = {
    "tomate": {
        "nombre_comun": "Tomate",
        "variedad": "Perita HD",
        "dias_totales": 90,
        "etapas_dias": [18, 27, 33, 12],  # Ini, Dev, Mid, Late
        "kc": [0.35, 0.70, 1.15, 0.75],
        "mad": [0.30, 0.40, 0.30, 0.55],   # MAD crítico en floración/cuaje = 0.30 -> theta_crit = 30 - 4.8 = 25.2%
        "temp_optima": (18.0, 25.0),
        "temp_helada": 0.0,
        "temp_estres": 32.0,
        "ph_optimo": (6.0, 6.8),
        "meses_siembra": [10, 11]  # Primavera: Oct-Nov (trasplante)
    },
    "ajo": {
        "nombre_comun": "Ajo",
        "variedad": "Colorado INTA",
        "dias_totales": 165,
        "etapas_dias": [25, 50, 68, 22],
        "kc": [0.45, 0.70, 1.15, 0.65],
        "mad": [0.30, 0.35, 0.25, 0.55],   # MAD crítico en bulbificación = 0.25 -> theta_crit = 30 - 4.0 = 26.0%
        "temp_optima": (12.0, 22.0),
        "temp_helada": -4.0,
        "temp_estres": 28.0,
        "ph_optimo": (6.2, 7.2),
        "meses_siembra": [3, 4]    # Otoño: Mar-Abr (directa)
    },
    "lechuga": {
        "nombre_comun": "Lechuga",
        "variedad": "Criolla / Grand Rapids",
        "dias_totales": 60,
        "etapas_dias": [12, 18, 22, 8],
        "kc": [0.45, 0.75, 1.00, 0.90],
        "mad": [0.25, 0.30, 0.20, 0.25],   # MAD más estricto = 0.20 -> theta_crit = 30 - 3.2 = 26.8%
        "temp_optima": (14.0, 19.0),
        "temp_helada": -2.0,
        "temp_estres": 26.0,
        "ph_optimo": (6.0, 7.0),
        "meses_siembra": [2, 3, 8, 9, 10]  # Casi todo el año
    },
    "cebolla": {
        "nombre_comun": "Cebolla",
        "variedad": "Valcatorce INTA",
        "dias_totales": 260,
        "etapas_dias": [38, 70, 118, 34],
        "kc": [0.40, 0.70, 1.10, 0.60],
        "mad": [0.25, 0.30, 0.25, 0.60],   # Secado fuerte pre-cosecha (mad 0.60 -> theta_crit = 20.4%)
        "temp_optima": (15.0, 25.0),
        "temp_helada": -3.0,
        "temp_estres": 32.0,
        "ph_optimo": (6.0, 7.0),
        "meses_siembra": [3, 4]    # Otoño: Mar-Abr
    },
    "pimiento": {
        "nombre_comun": "Pimiento",
        "variedad": "Calahorra / Trompa Elefante",
        "dias_totales": 90,
        "etapas_dias": [18, 27, 33, 12],
        "kc": [0.40, 0.70, 1.05, 0.80],
        "mad": [0.30, 0.35, 0.25, 0.40],   # Crítico en floración = 0.25 -> theta_crit = 26.0%
        "temp_optima": (20.0, 26.0),
        "temp_helada": 0.5,
        "temp_estres": 32.0,
        "ph_optimo": (6.0, 7.0),
        "meses_siembra": [10, 11]  # Primavera: trasplante Octubre
    },
    "papa": {
        "nombre_comun": "Papa",
        "variedad": "Spunta",
        "dias_totales": 120,
        "etapas_dias": [22, 28, 48, 22],
        "kc": [0.40, 0.75, 1.15, 0.70],
        "mad": [0.30, 0.35, 0.25, 0.50],   # Crítico en tuberización = 0.25 -> theta_crit = 26.0%
        "temp_optima": (16.0, 20.0),
        "temp_helada": -1.5,
        "temp_estres": 28.0,
        "ph_optimo": (5.2, 6.5),  # Evitar sarna común a pH > 6.8
        "meses_siembra": [9, 10, 11]  # Primavera temprana
    }
}

# Mapeo de features a sensores de la UI en Croply (PAN-AF-03)
# precip_pronostico_24h se mapea a "pronostico_lluvia_24h" para evitar contradicción con pluviómetro físico
SENSOR_UI_MAPPING = {
    "humedad_suelo_pct": "humedad_suelo_pct",
    "deficit_hidrico": "humedad_suelo_pct",
    "temp_ambiente_c": "temp_ambiente_c",
    "delta_t": "temp_ambiente_c",
    "viento_kmh": "viento_kmh",
    "wind_gust_kmh": "viento_kmh",
    "precip_pronostico_24h": "pronostico_lluvia_24h",
    "lluvia_mm_24h": "lluvia_mm_24h",
    "ph_suelo": "ph_suelo",
    "radiacion_solar_wm2": "radiacion_solar_wm2"
}

# Grupos semánticos para evitar factores repetidos en el Top 3 de explicabilidad SHAP
SEMANTIC_GROUPS = {
    "humedad_suelo_pct": "suelo_humedad",
    "deficit_hidrico": "suelo_humedad",
    "viento_kmh": "viento",
    "wind_gust_kmh": "viento",
    "temp_ambiente_c": "temperatura",
    "delta_t": "temperatura",
    "dias_desde_siembra": "fenologia",
    "etapa_fenologica_idx": "fenologia"
}

STATIC_NON_SENSOR_COLS = {
    "cultivo_enc",
    "dias_desde_siembra",
    "etapa_fenologica_idx",
    "hora_del_dia",
    "dias_desde_ultimo_riego",
    "dias_desde_ultima_curacion",
    "et0_diario"
}

# Features a excluir del Top 3 explicativo (solo variables físicas y agronómicas interpretables)
EXCLUDED_FROM_TOP_FACTORS = {
    "cultivo_enc",
    "dias_desde_siembra",
    "etapa_fenologica_idx",
    "hora_del_dia"
}


def calcular_delta_t(t_c: float, rh_pct: float) -> float:
    """Calcula el indicador psicrométrico Delta T según Stull (2011)."""
    rh = max(1.0, min(100.0, rh_pct))
    t = t_c
    tw = (t * math.atan(0.151977 * math.sqrt(rh + 8.313659)) +
          math.atan(t + rh) -
          math.atan(rh - 1.676331) +
          0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh) -
          4.686035)
    return round(max(0.0, t - tw), 2)


def get_etapa_fenologica(dias_desde_siembra: int, etapas_dias: List[int]) -> Tuple[int, str]:
    """Retorna índice de etapa (0: inicial, 1: desarrollo, 2: media, 3: final) y nombre."""
    cum_dias = 0
    nombres = ["inicial", "desarrollo", "media", "final"]
    for idx, d in enumerate(etapas_dias):
        cum_dias += d
        if dias_desde_siembra <= cum_dias:
            return idx, nombres[idx]
    return 3, "final"


def apply_feasibility_mask(proba_vector: np.ndarray, target_names: List[str], sample_dict: Dict[str, Any], cultivo_str: str, etapa_idx: int) -> np.ndarray:
    """
    Guardarraíl post-predicción para inferencia (HU-NA-03):
    Anula físicamente las probabilidades de acciones agronómicas imposibles según los sensores presentes.
    """
    proba = np.array(proba_vector, dtype=float).copy()
    
    val_ph = sample_dict.get("ph_suelo", np.nan)
    has_ph = (val_ph is not None and not pd.isna(val_ph))
    
    val_soil = sample_dict.get("humedad_suelo_pct", np.nan)
    has_soil = (val_soil is not None and not pd.isna(val_soil))
    
    precip_24h = sample_dict.get("precip_pronostico_24h", 0.0)
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


def infer_severity(pred_action: str, sample_dict: Dict[str, Any]) -> str:
    """Infiere la severidad ('Alta', 'Media', 'Baja') en tiempo de ejecución."""
    if pred_action == "REGAR_URGENTE":
        return "Alta"
    elif pred_action == "POSTERGAR_CURACION_CLIMA":
        viento = sample_dict.get("viento_kmh", 0.0)
        temp = sample_dict.get("temp_ambiente_c", 20.0)
        if (viento is not None and not pd.isna(viento) and viento >= 35.0) or (temp is not None and not pd.isna(temp) and temp >= 35.0):
            return "Alta"
        return "Media"
    elif pred_action == "SUSPENDER_RIEGO":
        precip = sample_dict.get("precip_pronostico_24h", 0.0)
        theta = sample_dict.get("humedad_suelo_pct", 0.0)
        if (precip is not None and not pd.isna(precip) and precip >= 30.0) or (theta is not None and not pd.isna(theta) and theta >= 33.0):
            return "Alta"
        return "Media"
    elif pred_action in ["CORREGIR_PH_SUELO", "REGAR_PROGRAMADO"]:
        return "Media"
    elif pred_action in ["VENTANA_OPTIMA_CURAR", "ESTADO_OPTIMO_SIN_ACCION"]:
        return "Baja"
    return "Media"


def get_dynamic_sensor_origin(feature_names: List[str], shap_values_for_class: np.ndarray, sample_dict: Dict[str, Any]) -> str:
    """Determina dinámicamente cuál fue el sensor físico real con mayor impacto positivo en la predicción."""
    best_feature = "humedad_suelo_pct"
    max_impact = -1e9
    
    # Primero buscar entre los que tienen SHAP > 0 (empujan a favor)
    for feat_name, shap_val in zip(feature_names, shap_values_for_class):
        if feat_name in STATIC_NON_SENSOR_COLS:
            continue
        val = sample_dict.get(feat_name, None)
        if val is None or pd.isna(val):
            continue
        if shap_val > max_impact:
            max_impact = shap_val
            best_feature = feat_name
            
    # Si ninguno fue > 0, tomar el de mayor magnitud absoluta
    if max_impact <= 0:
        max_abs = -1.0
        for feat_name, shap_val in zip(feature_names, shap_values_for_class):
            if feat_name in STATIC_NON_SENSOR_COLS:
                continue
            val = sample_dict.get(feat_name, None)
            if val is None or pd.isna(val):
                continue
            if abs(shap_val) > max_abs:
                max_abs = abs(shap_val)
                best_feature = feat_name
                
    return SENSOR_UI_MAPPING.get(best_feature, best_feature)


def get_deterministic_fallback(
    pred_action: str,
    inferred_severity: str,
    sensor_origin: str,
    sample_dict: Dict[str, Any],
    cultivo: str,
    etapa: str,
    alertas_activas: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Genera la respuesta determinista de respaldo (HU-NA-03) si Gemma 4 en Ollama
    experimenta timeout o fallas. Discrimina con precisión la causa de suspensión de riego
    y advierte sobre alertas meteorológicas incluso si el estado de suelo es óptimo.
    """
    cultivo_title = str(cultivo).capitalize()
    
    theta = sample_dict.get("humedad_suelo_pct", np.nan)
    theta_str = f"{theta:.1f}%" if (theta is not None and not pd.isna(theta)) else "sin telemetría directa"
    
    viento = sample_dict.get("viento_kmh", np.nan)
    viento_str = f"{viento:.1f} km/h" if (viento is not None and not pd.isna(viento)) else "velocidad no informada"
    
    ph = sample_dict.get("ph_suelo", np.nan)
    ph_str = f"{ph:.2f}" if (ph is not None and not pd.isna(ph)) else "no disponible"

    precip = sample_dict.get("precip_pronostico_24h", 0.0)
    precip_val = 0.0 if (precip is None or pd.isna(precip)) else float(precip)
    
    if pred_action == "REGAR_URGENTE":
        return {
            "titulo_corto": "Riego urgente recomendado",
            "texto_breve": f"Humedad del suelo ({theta_str}) bajo umbral crítico en {etapa}.",
            "texto_explicativo_extendido": (
                f"El cultivo de {cultivo_title} en etapa de {etapa} presenta un déficit hídrico severo detectado por {sensor_origin} ({theta_str}). "
                "Se aconseja activar inmediatamente el turno de riego para prevenir estrés irreversible y pérdida de rendimiento."
            )
        }
    elif pred_action == "REGAR_PROGRAMADO":
        return {
            "titulo_corto": "Riego preventivo programado",
            "texto_breve": f"Humedad ({theta_str}) próxima a agotar el agua útil.",
            "texto_explicativo_extendido": (
                f"La reserva hídrica de la parcela de {cultivo_title} está descendiendo hacia el umbral de reposición según pautas de INTA. "
                "Programe el turno de riego para las próximas 24 a 48 horas para sostener la capacidad de campo."
            )
        }
    elif pred_action == "SUSPENDER_RIEGO":
        # Discriminación agronómica explícita entre lluvia futura y saturación de suelo
        if sensor_origin == "pronostico_lluvia_24h" or precip_val >= 20.0:
            return {
                "titulo_corto": "Suspender riego por lluvias",
                "texto_breve": f"Lluvia abundante pronosticada ({precip_val:.1f} mm en 24h).",
                "texto_explicativo_extendido": (
                    f"Se pronostican precipitaciones significativas ({precip_val:.1f} mm) en la zona de {cultivo_title}. "
                    "Suspenda los turnos de riego planificados para evitar encharcamientos, asfixia radicular y desperdicio de agua."
                )
            }
        elif sensor_origin == "humedad_suelo_pct":
            return {
                "titulo_corto": "Suspender riego por saturación",
                "texto_breve": f"Humedad del suelo en exceso ({theta_str}).",
                "texto_explicativo_extendido": (
                    f"El sensor de humedad registra {theta_str}, superando la capacidad de campo para {cultivo_title}. "
                    "Se recomienda pausar el suministro hídrico para favorecer la oxigenación radicular y prevenir podredumbres fúngicas."
                )
            }
        else:
            return {
                "titulo_corto": "Suspender riego preventivo",
                "texto_breve": f"Etapa de precosecha o condiciones que contraindican el riego.",
                "texto_explicativo_extendido": (
                    f"El lote de {cultivo_title} en etapa de {etapa} requiere restricción hídrica para maduración y calidad de cosecha."
                )
            }
    elif pred_action == "VENTANA_OPTIMA_CURAR":
        return {
            "titulo_corto": "Ventana óptima para pulverizar",
            "texto_breve": f"Condiciones climáticas ideales: viento a {viento_str} y delta T apto.",
            "texto_explicativo_extendido": (
                f"Las condiciones meteorológicas son óptimas para realizar tratamientos fitosanitarios sobre {cultivo_title}. "
                f"El viento suave ({viento_str}) y la psicrometría aseguran máxima cobertura foliar con mínima deriva o evaporación."
            )
        }
    elif pred_action == "POSTERGAR_CURACION_CLIMA":
        return {
            "titulo_corto": "Postergar aplicación sanitaria",
            "texto_breve": f"Riesgo meteorológico adverso por variable {sensor_origin}.",
            "texto_explicativo_extendido": (
                f"Se desaconseja pulverizar sobre {cultivo_title} en este momento. "
                f"La lectura de {sensor_origin} está fuera del rango seguro, lo que generaría alta deriva o lavado prematuro del producto."
            )
        }
    elif pred_action == "CORREGIR_PH_SUELO":
        return {
            "titulo_corto": "Corrección de pH requerida",
            "texto_breve": f"pH del suelo ({ph_str}) desviado del rango óptimo para {cultivo_title}.",
            "texto_explicativo_extendido": (
                f"El sensor de suelo reporta un pH de {ph_str}, fuera del rango ideal para {cultivo_title}. "
                "Se sugiere evaluar enmiendas edáficas o acidificación en el agua de riego para optimizar la biodisponibilidad nutricional."
            )
        }
    else: # ESTADO_OPTIMO_SIN_ACCION
        if alertas_activas and len(alertas_activas) > 0:
            alerta_str = ", ".join(alertas_activas)
            alerta_principal = alertas_activas[0].replace("_", " ").title()
            return {
                "titulo_corto": f"Precaución: {alerta_principal}"[:40],
                "texto_breve": f"Suelo en rango óptimo, pero se detecta {alertas_activas[0].replace('_', ' ').lower()}."[:120],
                "texto_explicativo_extendido": (
                    f"El estado hídrico y nutricional del cultivo de {cultivo_title} en etapa de {etapa} es adecuado, "
                    f"pero las condiciones meteorológicas pronosticadas activaron {alerta_str}. "
                    "Se recomienda implementar medidas preventivas para proteger el lote."
                )
            }
        return {
            "titulo_corto": "Cultivo en condición óptima",
            "texto_breve": "Todos los parámetros edafoclimáticos dentro de los rangos meta.",
            "texto_explicativo_extendido": (
                f"El lote de {cultivo_title} en etapa de {etapa} vegeta en condiciones ideales de humedad y sanidad. "
                "Continúe con el esquema habitual de monitoreo."
            )
        }


class CroplyAnalyzer:
    """
    Singleton del motor analítico de Croply.
    Mantiene en memoria el clasificador LightGBM y el explainer SHAP.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CroplyAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return

        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "croply_classifier_v1.joblib")

        logging.info(f"[CroplyAnalyzer] Cargando artefactos del modelo desde: {model_path}...")
        t0 = time.time()
        artifacts = joblib.load(model_path)
        self.model = artifacts["model"]
        self.feature_cols = artifacts["feature_cols"]
        self.le_cultivo = artifacts["le_cultivo"]
        self.le_target = artifacts["le_target"]
        self.target_names = artifacts["target_names"]
        self.edaphic_constants = artifacts["edaphic_constants"]
        self.explainer = shap.TreeExplainer(self.model)
        
        # Warmup inicial de LightGBM y SHAP TreeExplainer
        dummy_row = pd.DataFrame([{col: 0.0 for col in self.feature_cols}])
        _ = self.model.predict_proba(dummy_row)
        _ = self.explainer.shap_values(dummy_row)
        
        self.version = "1.3.0-lgbm-loam-prod"
        self._initialized = True
        elapsed_ms = (time.time() - t0) * 1000
        logging.info(f"[CroplyAnalyzer] Inicialización y warmup completados en {elapsed_ms:.1f} ms. Inferencia lista.")

    def evaluate_climate_alerts(self, t_min: float, t_max: float, wind: float, gust: float, precip_24h: float, rh: float, temp: float, crop_cfg: dict, etapa_idx: int) -> List[Dict[str, str]]:
        """Evalúa las alertas climáticas automáticas según HU-NA-01."""
        alertas = []
        if t_min <= 0.0:
            alertas.append({
                "codigo_alerta": "ALERTA_HELADA",
                "severidad": "Alta",
                "descripcion": f"Temperatura mínima pronosticada de {t_min:.1f}°C (riesgo de helada meteorológica)."
            })
        elif t_min <= crop_cfg["temp_helada"] + 1.5 and etapa_idx in [0, 2]:
            alertas.append({
                "codigo_alerta": "ALERTA_HELADA_AGRONOMICA",
                "severidad": "Alta",
                "descripcion": f"Temperatura de {t_min:.1f}°C compromete tejidos sensibles en etapa fenológica crítica."
            })
            
        if precip_24h >= 30.0:
            alertas.append({
                "codigo_alerta": "ALERTA_LLUVIA_INTENSA",
                "severidad": "Media",
                "descripcion": f"Precipitación prevista de {precip_24h:.1f} mm en 24h (riesgo de anoxia y lavado)."
            })
            
        if wind >= 50.0 or gust >= 55.0 or (wind >= 35.0 and rh < 20.0 and temp > 28.0):
            alertas.append({
                "codigo_alerta": "ALERTA_VIENTO_FUERTE_ZONDA",
                "severidad": "Alta",
                "descripcion": f"Ráfagas de viento de {gust:.1f} km/h con condiciones compatibles con Viento Zonda."
            })
            
        if t_max >= 35.0:
            alertas.append({
                "codigo_alerta": "ALERTA_OLA_CALOR",
                "severidad": "Alta",
                "descripcion": f"Temperatura máxima pronosticada de {t_max:.1f}°C (estrés térmico y caída de flores)."
            })
            
        return alertas

    def analyze_parcel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis agronómico completo sobre la telemetría y pronóstico de una parcela.
        Mide el rendimiento granular de cada fase analítica con time.perf_counter().
        Latencia objetivo: < 5 ms.
        """
        t_start = time.perf_counter()
        
        cultivo_raw = str(payload.get("cultivo", "tomate")).lower()
        if cultivo_raw not in CROPS_CONFIG:
            raise ValueError(f"Cultivo '{cultivo_raw}' no soportado. Disponibles: {list(CROPS_CONFIG.keys())}")
        cfg = CROPS_CONFIG[cultivo_raw]
        
        dias_siembra = int(payload.get("dias_desde_siembra", 30))
        etapa_idx = payload.get("etapa_fenologica_idx")
        if etapa_idx is None:
            etapa_idx, etapa_nom = get_etapa_fenologica(dias_siembra, cfg["etapas_dias"])
        else:
            etapa_idx = int(etapa_idx)
            etapa_nom = ["inicial", "desarrollo", "media", "final"][min(3, max(0, etapa_idx))]

        mad_etapa = cfg["mad"][etapa_idx]
        theta_crit_etapa = THETA_CC - (mad_etapa * AU_TOTAL)

        # Extracción de telemetría y meteorología
        iot = payload.get("telemetria_iot", {})
        meteo = payload.get("pronostico_meteo", {})
        hist = payload.get("historial_manejo", {})
        
        t_min_futura = float(meteo.get("temp_min_pronostico", 12.0))
        t_max_futura = float(meteo.get("temp_max_pronostico", 28.0))
        precip_futura = float(meteo.get("precip_pronostico_24h", 0.0))
        
        # Fallback para sensor de temperatura ambiente desconectado o sin señal
        raw_temp_amb = iot.get("temp_ambiente_c")
        if raw_temp_amb is not None and not pd.isna(raw_temp_amb):
            temp_amb = float(raw_temp_amb)
        else:
            temp_amb = round((t_min_futura + t_max_futura) / 2.0, 1)

        # Fallback para sensor de humedad ambiente desconectado o sin señal
        raw_hum_amb = iot.get("humedad_ambiente_pct")
        if raw_hum_amb is not None and not pd.isna(raw_hum_amb):
            hum_amb = float(raw_hum_amb)
        else:
            hum_amb = 50.0

        theta_suelo = iot.get("humedad_suelo_pct")
        if theta_suelo is not None and not pd.isna(theta_suelo):
            theta_suelo = float(theta_suelo)
            deficit_hidrico = round(float(theta_crit_etapa - theta_suelo), 2)
        else:
            theta_suelo = np.nan
            deficit_hidrico = np.nan
            
        ph_suelo = iot.get("ph_suelo")
        ph_suelo = float(ph_suelo) if (ph_suelo is not None and not pd.isna(ph_suelo)) else np.nan
        
        rad_solar = iot.get("radiacion_solar_wm2")
        rad_solar = float(rad_solar) if (rad_solar is not None and not pd.isna(rad_solar)) else np.nan
        
        lluvia_pasada = float(iot.get("lluvia_mm_24h", 0.0))
        
        viento = float(meteo.get("viento_kmh", 10.0))
        gust = meteo.get("wind_gust_kmh")
        gust = float(gust) if gust is not None else round(viento * 1.25, 1)
        
        delta_t = calcular_delta_t(temp_amb, hum_amb)
        
        dias_riego = min(dias_siembra, int(hist.get("dias_desde_ultimo_riego", 4)))
        dias_curacion = min(dias_siembra, int(hist.get("dias_desde_ultima_curacion", 10)))
        
        cultivo_enc = int(self.le_cultivo.transform([cultivo_raw])[0])
        et0_diario = float(meteo.get("et0_diario", 5.0))
        hora_dia = int(payload.get("hora_del_dia", 12))
        
        # Vector de features para LightGBM
        feature_dict = {
            "cultivo_enc": cultivo_enc,
            "dias_desde_siembra": dias_siembra,
            "etapa_fenologica_idx": etapa_idx,
            "humedad_suelo_pct": theta_suelo,
            "temp_ambiente_c": temp_amb,
            "humedad_ambiente_pct": hum_amb,
            "radiacion_solar_wm2": rad_solar,
            "ph_suelo": ph_suelo,
            "lluvia_mm_24h": lluvia_pasada,
            "viento_kmh": viento,
            "wind_gust_kmh": gust,
            "temp_min_pronostico": t_min_futura,
            "temp_max_pronostico": t_max_futura,
            "precip_pronostico_24h": precip_futura,
            "et0_diario": et0_diario,
            "deficit_hidrico": deficit_hidrico,
            "delta_t": delta_t,
            "dias_desde_ultimo_riego": dias_riego,
            "dias_desde_ultima_curacion": dias_curacion,
            "hora_del_dia": hora_dia
        }
        df_x = pd.DataFrame([feature_dict])[self.feature_cols]
        t_prep_ms = (time.perf_counter() - t_start) * 1000
        
        # Inferencia con LightGBM + Guardarraíl Físico
        t_infer_start = time.perf_counter()
        proba_raw = self.model.predict_proba(df_x)[0]
        proba_masked = apply_feasibility_mask(proba_raw, self.target_names, feature_dict, cultivo_raw, etapa_idx)
        pred_idx = int(np.argmax(proba_masked))
        pred_action = self.target_names[pred_idx]
        confidence = float(proba_masked[pred_idx])
        severidad = infer_severity(pred_action, feature_dict)
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000
        
        # Explicabilidad SHAP local
        t_shap_start = time.perf_counter()
        shap_vals = self.explainer.shap_values(df_x)
        if isinstance(shap_vals, list):
            class_shap = shap_vals[pred_idx][0]
        elif len(shap_vals.shape) == 3:
            class_shap = shap_vals[0, :, pred_idx]
        else:
            class_shap = shap_vals[0]
            
        # Extracción Top 3 factores SHAP sin NaNs, con deduplicación y priorizando SHAP > 0
        indices_con_shap = list(range(len(self.feature_cols)))
        pos_indices = sorted([i for i in indices_con_shap if class_shap[i] > 0], key=lambda i: class_shap[i], reverse=True)
        neg_indices = sorted([i for i in indices_con_shap if class_shap[i] <= 0], key=lambda i: abs(class_shap[i]), reverse=True)
        sorted_indices = pos_indices + neg_indices

        top_features = []
        seen_groups = set()
        for idx_feat in sorted_indices:
            feat_name = self.feature_cols[idx_feat]
            # Excluir explícitamente variables no agronómicas / no físicas del Top 3
            if feat_name in EXCLUDED_FROM_TOP_FACTORS:
                continue
            val = df_x.iloc[0][feat_name]
            if pd.isna(val):
                continue
            group = SEMANTIC_GROUPS.get(feat_name, feat_name)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            top_features.append({
                "feature": feat_name,
                "sensor_ui": SENSOR_UI_MAPPING.get(feat_name, feat_name),
                "valor": float(val),
                "shap_attribution": round(float(class_shap[idx_feat]), 4)
            })
            if len(top_features) == 3:
                break
                
        sensor_origen = get_dynamic_sensor_origin(self.feature_cols, class_shap, feature_dict)
        t_shap_ms = (time.perf_counter() - t_shap_start) * 1000
        
        # Alertas climáticas automáticas (HU-NA-01) y Fallback Determinista
        t_rules_start = time.perf_counter()
        alertas = self.evaluate_climate_alerts(
            t_min_futura, t_max_futura, viento, gust, precip_futura, hum_amb, temp_amb, cfg, etapa_idx
        )
        codigos_alertas = [a["codigo_alerta"] for a in alertas]
        has_real_alert = len(alertas) > 0
        
        # Condición de activación de LLM (HU-NA-03)
        requiere_intervencion_llm = bool(pred_action != "ESTADO_OPTIMO_SIN_ACCION" or has_real_alert)
        
        # Fallback determinista (considerando alertas activas si la parcela está en estado óptimo)
        fallback = get_deterministic_fallback(
            pred_action, severidad, sensor_origen, feature_dict, cultivo_raw, etapa_nom, codigos_alertas
        )
        t_rules_ms = (time.perf_counter() - t_rules_start) * 1000
        
        t_total_analyzer_ms = (time.perf_counter() - t_start) * 1000
        
        return {
            "metadata": {
                "version_analyzer": self.version,
                "parcela_id": payload.get("parcela_id", "parcela-default"),
                "cultivo": cultivo_raw,
                "etapa_fenologica": etapa_nom,
                "etapa_fenologica_idx": etapa_idx,
                "dias_desde_siembra": dias_siembra,
                "timestamp": payload.get("timestamp_utc", "2026-10-17T18:30:00Z"),
                "latencia_analyzer_ms": round(t_total_analyzer_ms, 2),
                "desglose_latencia_analyzer": {
                    "preprocesamiento_features_ms": round(t_prep_ms, 3),
                    "inferencia_lightgbm_ms": round(t_infer_ms, 3),
                    "explicabilidad_shap_ms": round(t_shap_ms, 3),
                    "reglas_climaticas_fallback_ms": round(t_rules_ms, 3)
                }
            },
            "prediccion_ml": {
                "accion_recomendada": pred_action,
                "severidad": severidad,
                "confianza": round(confidence, 4),
                "sensor_origen_principal": sensor_origen,
                "requiere_intervencion_llm": requiere_intervencion_llm,
                "distribucion_probabilidad": {
                    self.target_names[i]: round(float(proba_masked[i]), 4)
                    for i in range(len(self.target_names))
                },
                "explicabilidad_top3_factores": top_features
            },
            "response_schema_esperado": {
                "tipo": "object",
                "campos_requeridos": ["titulo_corto", "texto_breve", "texto_explicativo_extendido"],
                "especificaciones": {
                    "titulo_corto": "Título imperativo/badge, máximo 40 caracteres.",
                    "texto_breve": "Resumen directo con datos del sensor principal, máximo 120 caracteres.",
                    "texto_explicativo_extendido": "Explicación agronómica y empática de 2 a 3 oraciones contextualizada a la etapa fenológica."
                }
            },
            "fallback_determinista": fallback,
            "alertas_climaticas_activas": codigos_alertas,
            "alertas_climaticas_detalle": alertas,
            "riesgo_climatico_fenologico": {
                "nivel": "ALTO" if has_real_alert else "BAJO",
                "impacto": f"Condición climática evaluada con sensibilidad de etapa {etapa_nom} ({cfg['nombre_comun']})"
            },
            "telemetria_actual": {
                "humedad_suelo_pct": None if pd.isna(theta_suelo) else float(theta_suelo),
                "temp_ambiente_c": float(temp_amb),
                "humedad_ambiente_pct": float(hum_amb),
                "radiacion_solar_wm2": None if pd.isna(rad_solar) else float(rad_solar),
                "ph_suelo": None if pd.isna(ph_suelo) else float(ph_suelo),
                "viento_kmh": float(viento),
                "wind_gust_kmh": float(gust),
                "delta_t_psicrometrico": float(delta_t),
                "precip_pronostico_24h": float(precip_futura),
                "deficit_hidrico_pct": None if pd.isna(deficit_hidrico) else float(deficit_hidrico)
            },
            "umbrales_agronomicos_referencia_inta": {
                "suelo_tipo": "Franco",
                "capacidad_campo_pct": THETA_CC,
                "punto_marchitez_pct": THETA_PMP,
                "agua_util_pct": AU_TOTAL,
                "rango_ph_optimo": cfg["ph_optimo"],
                "viento_maximo_pulverizacion_kmh": 15.0,
                "delta_t_optimo_pulverizacion": [2.0, 8.0]
            }
        }


def get_analyzer() -> CroplyAnalyzer:
    """Función de acceso al singleton del analizador."""
    return CroplyAnalyzer()
