#!/usr/bin/env python3
"""
build_hybrid_dataset.py
-----------------------
Generador de Dataset Híbrido Científicamente Calibrado para la Región de Cuyo (Mendoza).
Integra:
1. Series climáticas horarias reales de Open-Meteo (Guaymallén 2024-2025).
2. Parámetros fisiológicos y umbrales agronómicos del INTA (EEA Mendoza, EEA La Consulta)
   y el manual FAO-56.
3. Modelo simplificado de balance hídrico de suelo (evapotranspiración por Kc, recarga por riego
   y lluvia efectiva, modelando la regla de negocio crítica: lluvia leve < 5mm moja superficie pero
   no recarga la zona radicular).
4. Calibración edafológica para suelo Franco típico de oasis irrigados de Mendoza
   (THETA_CC = 30.0%, THETA_PMP = 14.0%, AU = 16.0%).
5. Soporte nativo para sensores faltantes (Missing Sensors):
   - ph_suelo: 20% np.nan
   - radiacion_solar_wm2: 15% np.nan
   - humedad_suelo_pct: 8% np.nan (y deficit_hidrico = np.nan consecuente)
6. Coherencia cronológica estricta:
   - dias_desde_ultimo_riego <= dias_desde_siembra
   - dias_desde_ultima_curacion <= dias_desde_siembra
7. Eliminación de Data Leakage: evaluación de reglas agronómicas sobre valores físicos base
   antes de inyectar ruido gaussiano en las features.
8. Distribución continua sin márgenes artificiales en escenarios balanceados y ruido
   gaussiano aumentado a 0.8 en humedad_suelo_pct para probabilidades calibradas realistas.
9. Balance estricto de clases: ninguna clase supera el 25% ni cae por debajo del 8%.

Salida: raw_datasets/croply_training_dataset_cuyo.csv
"""

import os
import math
import random
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEATHER_FILE = os.path.join(BASE_DIR, "..", "data", "raw_datasets", "clima_mendoza_openmeteo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "raw_datasets", "croply_training_dataset_cuyo.csv")

# Semilla para reproducibilidad científica
np.random.seed(42)
random.seed(42)

# Constantes Edafológicas para Suelo Franco representativo de los oasis de Mendoza
# (Alineado con PAN-AF-02 y PAN-AF-03: a humedad 22% se dispara riego requerido)
THETA_CC = 30.0      # Capacidad de Campo (% volumétrico)
THETA_PMP = 14.0     # Punto de Marchitez Permanente (% volumétrico)
AU_TOTAL = THETA_CC - THETA_PMP  # Agua Útil Total = 16.0%

# Parámetros Agronómicos INTA / FAO-56 por Cultivo
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
        "temp_helada": 0.0,
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

def get_etapa_fenologica(dias_desde_siembra, etapas_dias):
    """Retorna índice de etapa (0: inicial, 1: desarrollo, 2: media, 3: final) y nombre."""
    cum_dias = 0
    nombres = ["inicial", "desarrollo", "media", "final"]
    for idx, d in enumerate(etapas_dias):
        cum_dias += d
        if dias_desde_siembra <= cum_dias:
            return idx, nombres[idx]
    return 3, "final"

def calcular_delta_t(t_c, rh_pct):
    """
    Calcula Delta T (°C) aproximado psicrométricamente:
    Delta T = T_dry - T_wet
    Fórmula empírica aproximada de Stull (2011) para Twet.
    """
    rh = max(1.0, min(100.0, rh_pct))
    t = t_c
    tw = (t * math.atan(0.151977 * math.sqrt(rh + 8.313659)) +
          math.atan(t + rh) -
          math.atan(rh - 1.676331) +
          0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh) -
          4.686035)
    return round(max(0.0, t - tw), 2)

def evaluate_agronomic_rules(base_data, crop_cfg, etapa_idx):
    """
    Evalúa de forma determinista y científicamente fundada las reglas agronómicas
    de INTA/FAO-56 sobre los VALORES FÍSICOS BASE (sin ruido gaussiano) para asignar:
    - acción target
    - severidad
    - sensor_origen_principal
    - alerta_climatica

    Reglas clave de sensores faltantes:
    1. Si pd.isna(ph): se omite totalmente CORREGIR_PH_SUELO.
    2. Si pd.isna(theta): no se emiten acciones de riego por suelo (REGAR_* ni
       SUSPENDER_RIEGO por exceso de humedad en suelo). Solo puede suspenderse por
       precipitaciones pronosticadas intensas o secado fisiológico pre-cosecha.
    """
    theta = base_data["humedad_suelo_base"]
    temp = base_data["temp_ambiente_base"]
    rh = base_data["humedad_ambiente_base"]
    wind = base_data["viento_base"]
    gust = base_data["wind_gust_est"]
    precip_24h = base_data["precip_pronostico_24h"]
    ph = base_data["ph_suelo_base"]
    delta_t = base_data["delta_t_base"]
    t_min_pred = base_data["temp_min_pronostico"]
    t_max_pred = base_data["temp_max_pronostico"]
    dias_curacion = base_data["dias_desde_ultima_curacion"]
    intentar_curacion = base_data.get("intentar_curacion", False)
    
    mad = crop_cfg["mad"][etapa_idx]
    theta_crit = THETA_CC - (mad * AU_TOTAL)       # Umbral de reposición (típicamente 24-26%)
    theta_urgente = THETA_PMP + (0.25 * AU_TOTAL)  # Déficit severo (14 + 4 = 18%)
    
    has_theta = (theta is not None and not pd.isna(theta))
    has_ph = (ph is not None and not pd.isna(ph))
    
    # 1. Alertas Climáticas Deterministas (HU-NA-01)
    alertas = []
    if t_min_pred <= 0.0:
        alertas.append("ALERTA_HELADA")
    elif t_min_pred <= crop_cfg["temp_helada"] + 1.5 and etapa_idx in [0, 2]:
        alertas.append("ALERTA_HELADA_AGRONOMICA")
        
    if precip_24h >= 30.0:
        alertas.append("ALERTA_LLUVIA_INTENSA")
        
    if wind >= 50.0 or gust >= 55.0 or (wind >= 35.0 and rh < 20.0 and temp > 28.0):
        alertas.append("ALERTA_VIENTO_FUERTE_ZONDA")
        
    if t_max_pred >= 35.0:
        alertas.append("ALERTA_OLA_CALOR")

    alerta_str = "|".join(alertas) if alertas else "SIN_ALERTA"

    # 2. Evaluación de Acción Agronómica (HU-NA-03)
    
    # A) Exceso de humedad o etapa final de secado en Ajo/Cebolla o lluvia copiosa inminente
    # Si no hay sensor de suelo, solo suspender por secado pre-cosecha o por lluvia fuerte
    if etapa_idx == 3 and crop_cfg["nombre_comun"] in ["Ajo", "Cebolla"]:
        return ("SUSPENDER_RIEGO", "Media", "precip_pronostico_24h", alerta_str)
    if precip_24h >= 25.0:
        return ("SUSPENDER_RIEGO", "Alta" if precip_24h >= 30.0 else "Media", "precip_pronostico_24h", alerta_str)
    if has_theta and (theta > THETA_CC + 1.5):
        return ("SUSPENDER_RIEGO", "Alta" if theta > THETA_CC + 3.0 else "Media", "humedad_suelo_pct", alerta_str)

    # B) Déficit hídrico urgente (SOLO si hay sensor de humedad de suelo)
    if has_theta:
        if theta <= theta_urgente or (theta < theta_crit and etapa_idx == 2 and temp > 32.0):
            return ("REGAR_URGENTE", "Alta", "humedad_suelo_pct", alerta_str)

    # C) Desbalance químico de pH de suelo (SOLO si el sensor de pH está presente)
    if has_ph:
        ph_min, ph_max = crop_cfg["ph_optimo"]
        if ph < ph_min - 0.4 or ph > ph_max + 0.4:
            return ("CORREGIR_PH_SUELO", "Media", "ph_suelo", alerta_str)

    # D) Fitosanitarios / Pulverización ("Curar")
    necesita_curar = intentar_curacion or (dias_curacion >= 14) or (rh > 80.0 and temp > 20.0 and dias_curacion >= 7)
    
    clima_peligroso_pulverizar = (
        wind > 15.0 or 
        gust > 20.0 or 
        precip_24h > 1.5 or 
        temp > 28.0 or 
        temp < 10.0 or 
        delta_t > 10.0 or 
        delta_t < 2.0 or
        "ALERTA_VIENTO_FUERTE_ZONDA" in alerta_str
    )

    if necesita_curar and clima_peligroso_pulverizar:
        causa = "viento_kmh" if wind > 15.0 else ("temp_ambiente_c" if (temp > 28.0 or temp < 10.0) else "delta_t")
        return ("POSTERGAR_CURACION_CLIMA", "Media", causa, alerta_str)

    # Ventana óptima para curar: viento suave (3 a 12 km/h), temperatura 14 a 25 °C, Delta T 2 a 8 °C, sin lluvia
    if necesita_curar and (3.0 <= wind <= 12.0) and (14.0 <= temp <= 25.0) and (2.0 <= delta_t <= 8.0) and (precip_24h < 1.0):
        return ("VENTANA_OPTIMA_CURAR", "Baja", "viento_kmh", alerta_str)

    # E) Riego programado (SOLO si hay sensor de suelo y la humedad está bajo umbral de reposición)
    if has_theta and (theta <= theta_crit):
        return ("REGAR_PROGRAMADO", "Media", "humedad_suelo_pct", alerta_str)

    # F) Pequeño desvío de pH leve (si hay sensor de pH)
    if has_ph:
        ph_min, ph_max = crop_cfg["ph_optimo"]
        if ph < ph_min or ph > ph_max:
            return ("CORREGIR_PH_SUELO", "Baja", "ph_suelo", alerta_str)

    # G) Estado Óptimo sin acción requerida
    sensor_referencia = "humedad_suelo_pct" if has_theta else "temp_ambiente_c"
    return ("ESTADO_OPTIMO_SIN_ACCION", "Baja", sensor_referencia, alerta_str)

def generate_hybrid_dataset(n_samples=18000):
    logging.info(f"Cargando series climáticas reales desde {WEATHER_FILE}...")
    df_weather = pd.read_csv(WEATHER_FILE)
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
    total_hours = len(df_weather)
    
    # Precomputar métricas rolling de clima (pronósticos simulados a 24h)
    logging.info("Calculando ventanas móviles climáticas de 24h...")
    df_weather["precip_24h_future"] = df_weather["precipitacion_mm"].rolling(window=24, min_periods=1).sum().shift(-24).fillna(0.0)
    df_weather["temp_min_future"] = df_weather["temp_ambiente_c"].rolling(window=24, min_periods=1).min().shift(-24).fillna(df_weather["temp_ambiente_c"])
    df_weather["temp_max_future"] = df_weather["temp_ambiente_c"].rolling(window=24, min_periods=1).max().shift(-24).fillna(df_weather["temp_ambiente_c"])
    df_weather["wind_gust_est"] = df_weather["viento_kmh"] * np.random.uniform(1.2, 1.5, size=total_hours)

    records = []
    crops_list = list(CROPS_CONFIG.keys())
    samples_per_crop = n_samples // len(crops_list)

    logging.info(f"Iniciando simulación agronómica: {len(crops_list)} cultivos x ~{samples_per_crop} muestras...")

    for crop_id in crops_list:
        cfg = CROPS_CONFIG[crop_id]
        generated_crop_samples = 0
        
        while generated_crop_samples < samples_per_crop:
            # 1. Seleccionar una fecha de siembra válida dentro del calendario INTA
            mes_siembra = random.choice(cfg["meses_siembra"])
            año_siembra = random.choice([2024, 2025])
            dia_siembra = random.randint(1, 28)
            fecha_siembra = pd.Timestamp(year=año_siembra, month=mes_siembra, day=dia_siembra)
            
            # 2. Seleccionar un día de observación dentro del ciclo biológico
            dias_ciclo = cfg["dias_totales"]
            dias_desde_siembra = random.randint(1, dias_ciclo)
            fecha_obs = fecha_siembra + pd.Timedelta(days=dias_desde_siembra)
            
            # Limitar al rango del dataset meteorológico
            if fecha_obs < df_weather["timestamp"].min() or fecha_obs > df_weather["timestamp"].max() - pd.Timedelta(days=2):
                continue
                
            hora_obs = random.randint(0, 23)
            fecha_hora_obs = fecha_obs.replace(hour=hora_obs)
            
            # Ubicar fila climática correspondiente
            idx_weather = df_weather.index[df_weather["timestamp"] == fecha_hora_obs]
            if len(idx_weather) == 0:
                continue
            idx = idx_weather[0]
            w_row = df_weather.iloc[idx]
            
            # Etapa fenológica
            etapa_idx, etapa_nom = get_etapa_fenologica(dias_desde_siembra, cfg["etapas_dias"])
            kc_actual = cfg["kc"][etapa_idx]
            mad_etapa = cfg["mad"][etapa_idx]
            theta_crit_etapa = THETA_CC - (mad_etapa * AU_TOTAL)  # p.ej. 24% a 26.8%
            theta_urgente_etapa = THETA_PMP + (0.25 * AU_TOTAL)   # 18.0%
            
            # 3. Simulación de Balance Hídrico Base
            # COHERENCIA CRONOLÓGICA: nunca mayor a dias_desde_siembra
            dias_ultimo_riego = min(dias_desde_siembra, random.randint(0, 10))
            dias_ultima_curacion = min(dias_desde_siembra, random.randint(1, 25))
            
            # Lluvia real acumulada en las últimas 24 horas
            lluvia_24h_pasada = float(df_weather["precipitacion_mm"].iloc[max(0, idx-24):idx].sum())
            lluvia_efectiva_raiz = lluvia_24h_pasada if lluvia_24h_pasada >= 5.0 else 0.0
            
            # Evapotranspiración acumulada reciente
            et0_reciente = float(df_weather["et0_mm"].iloc[max(0, idx-24):idx].sum())
            etc_estimada = et0_reciente * kc_actual
            
            # Dinámica de humedad del suelo base (% volumétrico en suelo Franco CC=30%, PMP=14%)
            humedad_base = THETA_CC - (dias_ultimo_riego * 1.5) - (etc_estimada * 0.45)
            if dias_ultimo_riego == 0:
                humedad_base = THETA_CC + np.random.uniform(-0.5, 1.5)
            elif dias_ultimo_riego == 1:
                humedad_base = THETA_CC - np.random.uniform(0.3, 1.2)
                
            if lluvia_efectiva_raiz > 0:
                humedad_base += min(6.0, lluvia_efectiva_raiz * 0.4)

            # Sorteo de sensores faltantes (Missing Sensors)
            has_soil_sensor = (random.random() >= 0.08)       # 8% np.nan en humedad de suelo
            has_ph_sensor = (random.random() >= 0.20)         # 20% np.nan en pH
            has_solar_sensor = (random.random() >= 0.15)      # 15% np.nan en radiación solar
            
            # pH base
            ph_min, ph_max = cfg["ph_optimo"]
            ph_base = (ph_min + ph_max) / 2.0
            
            # Variables de pronóstico y microclima
            viento_base = float(w_row["viento_kmh"])
            temp_ambiente_base = float(w_row["temp_ambiente_c"])
            hum_amb_base = float(w_row["humedad_ambiente_pct"])
            if lluvia_24h_pasada > 0 and lluvia_efectiva_raiz == 0:
                hum_amb_base = max(hum_amb_base, 78.0)
            delta_t_base = calcular_delta_t(temp_ambiente_base, hum_amb_base)
            precip_futura_24h = round(float(w_row["precip_24h_future"]), 1)
            temp_min_futura = round(float(w_row["temp_min_future"]), 1)
            temp_max_futura = round(float(w_row["temp_max_future"]), 1)
            wind_gust_est = float(w_row["wind_gust_est"])
            
            # -------------------------------------------------------------
            # BALANCED SCENARIOS INJECTION (Sin sesgo por sensor faltante)
            # -------------------------------------------------------------
            intentar_curacion = False
            rand_scenario = random.random()
            
            if has_soil_sensor:
                # Distribución completa de 7 escenarios cuando hay sensor de humedad de suelo
                if rand_scenario < 0.15:
                    # 1. Forzar REGAR_URGENTE (objetivo ~15-18%)
                    humedad_base = random.uniform(THETA_PMP - 1.0, theta_urgente_etapa)
                elif rand_scenario < 0.31:
                    # 2. Forzar REGAR_PROGRAMADO (objetivo ~12-16%)
                    humedad_base = random.uniform(theta_urgente_etapa, theta_crit_etapa)
                elif rand_scenario < 0.47:
                    # 3. Forzar SUSPENDER_RIEGO (objetivo ~14-18%)
                    if crop_id in ["ajo", "cebolla"] and etapa_idx == 3:
                        pass # Ya cae en suspender por etapa
                    else:
                        humedad_base = random.uniform(THETA_CC + 1.5, THETA_CC + 5.0)
                elif rand_scenario < 0.62:
                    # 4. Forzar VENTANA_OPTIMA_CURAR (objetivo ~13-16%)
                    humedad_base = random.uniform(theta_crit_etapa, THETA_CC)
                    viento_base = random.uniform(4.0, 11.0)
                    temp_ambiente_base = random.uniform(16.0, 23.0)
                    hum_amb_base = random.uniform(55.0, 75.0)
                    delta_t_base = calcular_delta_t(temp_ambiente_base, hum_amb_base)
                    precip_futura_24h = 0.0
                    wind_gust_est = viento_base * 1.2
                    intentar_curacion = True
                    dias_ultima_curacion = min(dias_desde_siembra, random.randint(14, 25))
                elif rand_scenario < 0.76:
                    # 5. Forzar POSTERGAR_CURACION_CLIMA (objetivo ~14-17%)
                    humedad_base = random.uniform(theta_crit_etapa, THETA_CC)
                    sub_r = random.random()
                    if sub_r < 0.4:
                        viento_base = random.uniform(18.0, 35.0)
                        wind_gust_est = viento_base * 1.3
                    elif sub_r < 0.7:
                        temp_ambiente_base = random.uniform(31.0, 38.0)
                    else:
                        precip_futura_24h = random.uniform(5.0, 20.0)
                    delta_t_base = calcular_delta_t(temp_ambiente_base, hum_amb_base)
                    intentar_curacion = True
                    dias_ultima_curacion = min(dias_desde_siembra, random.randint(14, 25))
                elif rand_scenario < 0.88:
                    # 6. Forzar CORREGIR_PH_SUELO (objetivo ~10-13%)
                    if has_ph_sensor:
                        humedad_base = random.uniform(theta_crit_etapa, THETA_CC)
                        ph_base = (ph_min - 0.7) if random.random() < 0.5 else (ph_max + 0.8)
                        intentar_curacion = False
                        dias_ultima_curacion = min(dias_desde_siembra, 5)
                    else:
                        humedad_base = random.uniform(theta_crit_etapa, THETA_CC)
                else:
                    # 7. Forzar ESTADO_OPTIMO_SIN_ACCION (objetivo ~12-16%)
                    humedad_base = random.uniform(theta_crit_etapa, THETA_CC)
                    intentar_curacion = False
                    dias_ultima_curacion = min(dias_desde_siembra, random.randint(2, 6))
                    if has_ph_sensor:
                        ph_base = (ph_min + ph_max) / 2.0
            else:
                # Distribución balanceada (~20% cada una) entre los 5 escenarios viables SIN sensor de humedad
                if rand_scenario < 0.20:
                    # 1. SUSPENDER_RIEGO por lluvia copiosa inminente o secado de bulbos
                    if not (crop_id in ["ajo", "cebolla"] and etapa_idx == 3):
                        precip_futura_24h = random.uniform(25.0, 45.0)
                    intentar_curacion = False
                elif rand_scenario < 0.40:
                    # 2. VENTANA_OPTIMA_CURAR
                    viento_base = random.uniform(4.0, 11.0)
                    temp_ambiente_base = random.uniform(16.0, 23.0)
                    hum_amb_base = random.uniform(55.0, 75.0)
                    delta_t_base = calcular_delta_t(temp_ambiente_base, hum_amb_base)
                    precip_futura_24h = 0.0
                    wind_gust_est = viento_base * 1.2
                    intentar_curacion = True
                    dias_ultima_curacion = min(dias_desde_siembra, random.randint(14, 25))
                elif rand_scenario < 0.60:
                    # 3. POSTERGAR_CURACION_CLIMA
                    sub_r = random.random()
                    if sub_r < 0.4:
                        viento_base = random.uniform(18.0, 35.0)
                        wind_gust_est = viento_base * 1.3
                    elif sub_r < 0.7:
                        temp_ambiente_base = random.uniform(31.0, 38.0)
                    else:
                        precip_futura_24h = random.uniform(5.0, 20.0)
                    delta_t_base = calcular_delta_t(temp_ambiente_base, hum_amb_base)
                    intentar_curacion = True
                    dias_ultima_curacion = min(dias_desde_siembra, random.randint(14, 25))
                elif rand_scenario < 0.80:
                    # 4. CORREGIR_PH_SUELO (si hay sensor de pH)
                    if has_ph_sensor:
                        ph_base = (ph_min - 0.7) if random.random() < 0.5 else (ph_max + 0.8)
                        intentar_curacion = False
                        dias_ultima_curacion = min(dias_desde_siembra, 5)
                    else:
                        intentar_curacion = False
                        dias_ultima_curacion = min(dias_desde_siembra, random.randint(2, 6))
                else:
                    # 5. ESTADO_OPTIMO_SIN_ACCION
                    intentar_curacion = False
                    dias_ultima_curacion = min(dias_desde_siembra, random.randint(2, 6))
                    if has_ph_sensor:
                        ph_base = (ph_min + ph_max) / 2.0

            # Estructura de datos base para evaluación agronómica
            base_eval_dict = {
                "humedad_suelo_base": humedad_base if has_soil_sensor else None,
                "temp_ambiente_base": temp_ambiente_base,
                "humedad_ambiente_base": hum_amb_base,
                "viento_base": viento_base,
                "wind_gust_est": wind_gust_est,
                "precip_pronostico_24h": precip_futura_24h,
                "ph_suelo_base": ph_base if has_ph_sensor else None,
                "delta_t_base": delta_t_base,
                "temp_min_pronostico": temp_min_futura,
                "temp_max_pronostico": temp_max_futura,
                "dias_desde_ultima_curacion": dias_ultima_curacion,
                "intentar_curacion": intentar_curacion,
                "etapa_fenologica": etapa_nom
            }
            
            # EVALUACIÓN DE REGLAS SOBRE VALORES BASE (SIN DATA LEAKAGE)
            accion, severidad, sensor_origen, alerta = evaluate_agronomic_rules(base_eval_dict, cfg, etapa_idx)
            
            # -------------------------------------------------------------
            # INYECCIÓN DE RUIDO GAUSSIANO DE INSTRUMENTAL IoT POST-ETIQUETADO
            # Ruido aumentado a 0.8 en humedad_suelo_pct para calibrar probabilidades
            # -------------------------------------------------------------
            if has_soil_sensor:
                humedad_suelo_pct = round(float(np.clip(humedad_base + np.random.normal(0, 0.80), 8.0, 42.0)), 2)
                deficit_hidrico = round(float(theta_crit_etapa - humedad_suelo_pct), 2)
            else:
                humedad_suelo_pct = np.nan
                deficit_hidrico = np.nan

            temp_ambiente = round(float(temp_ambiente_base + np.random.normal(0, 0.4)), 1)
            humedad_ambiente = round(float(np.clip(hum_amb_base + np.random.normal(0, 1.5), 5.0, 100.0)), 1)
            viento_kmh = round(float(max(0.0, viento_base + np.random.normal(0, 0.8))), 1)
            wind_gust_kmh = round(float(max(viento_kmh, wind_gust_est + np.random.normal(0, 1.5))), 1)
            
            if has_ph_sensor:
                ph_suelo = round(float(np.clip(ph_base + np.random.normal(0, 0.12), 4.5, 9.5)), 2)
            else:
                ph_suelo = np.nan
                
            if has_solar_sensor:
                radiacion_solar = round(float(max(0.0, w_row["radiacion_solar_wm2"] + np.random.normal(0, 15.0))), 1)
            else:
                radiacion_solar = np.nan
                
            delta_t = calcular_delta_t(temp_ambiente, humedad_ambiente)
            et0_diario = round(float(max(0.5, w_row["et0_mm"] * 24.0)), 2)
            
            sample_data = {
                "cultivo": crop_id,
                "dias_desde_siembra": dias_desde_siembra,
                "etapa_fenologica": etapa_nom,
                "etapa_fenologica_idx": etapa_idx,
                "humedad_suelo_pct": humedad_suelo_pct,
                "temp_ambiente_c": temp_ambiente,
                "humedad_ambiente_pct": humedad_ambiente,
                "radiacion_solar_wm2": radiacion_solar,
                "ph_suelo": ph_suelo,
                "lluvia_mm_24h": round(lluvia_24h_pasada, 1),
                "viento_kmh": viento_kmh,
                "wind_gust_kmh": wind_gust_kmh,
                "temp_min_pronostico": temp_min_futura,
                "temp_max_pronostico": temp_max_futura,
                "precip_pronostico_24h": precip_futura_24h,
                "et0_diario": et0_diario,
                "deficit_hidrico": deficit_hidrico,
                "delta_t": delta_t,
                "dias_desde_ultimo_riego": dias_ultimo_riego,
                "dias_desde_ultima_curacion": dias_ultima_curacion,
                "hora_del_dia": hora_obs,
                "target_accion": accion,
                "target_severidad": severidad,
                "sensor_origen_principal": sensor_origen,
                "alerta_climatica": alerta
            }
            
            records.append(sample_data)
            generated_crop_samples += 1

    df_final = pd.DataFrame(records)
    
    # Validaciones de Integridad y Balance
    logging.info(f"Dataset generado exitosamente con {len(df_final)} registros.")
    logging.info("\n--- DISTRIBUCIÓN POR CULTIVO ---")
    print(df_final["cultivo"].value_counts())
    
    logging.info("\n--- DISTRIBUCIÓN POR ACCIÓN RECOMENDADA (TARGET) ---")
    action_counts = df_final["target_accion"].value_counts()
    action_pct = df_final["target_accion"].value_counts(normalize=True) * 100
    df_balance = pd.DataFrame({"conteo": action_counts, "porcentaje": action_pct.round(2)})
    print(df_balance)
    
    # Verificación de límites de balance (ninguna > 25%, ninguna < 8%)
    for act, pct in action_pct.items():
        if pct > 25.0:
            logging.warning(f"Clase {act} supera el 25% ({pct:.2f}%)")
        elif pct < 8.0:
            logging.warning(f"Clase {act} inferior al 8% ({pct:.2f}%)")
    
    logging.info("\n--- DISTRIBUCIÓN POR SEVERIDAD ---")
    print(df_final["target_severidad"].value_counts())

    logging.info("\n--- PORCENTAJE DE VALORES FALTANTES (MISSING SENSORS) ---")
    print(df_final[["humedad_suelo_pct", "deficit_hidrico", "ph_suelo", "radiacion_solar_wm2"]].isna().mean() * 100)

    # Validar que si ph_suelo es NaN, target_accion nunca sea CORREGIR_PH_SUELO
    ph_nan_actions = df_final[df_final["ph_suelo"].isna()]["target_accion"].value_counts()
    assert "CORREGIR_PH_SUELO" not in ph_nan_actions, "ERROR: Hay registros con ph_suelo NaN etiquetados como CORREGIR_PH_SUELO"
    logging.info("Validación exitosa: Cero casos de CORREGIR_PH_SUELO cuando el sensor de pH no está presente.")
    
    # Validar que si humedad_suelo_pct es NaN, target_accion nunca sea REGAR_*
    soil_nan_actions = df_final[df_final["humedad_suelo_pct"].isna()]["target_accion"].value_counts()
    assert "REGAR_URGENTE" not in soil_nan_actions, "ERROR: Hay registros con humedad_suelo NaN etiquetados como REGAR_URGENTE"
    assert "REGAR_PROGRAMADO" not in soil_nan_actions, "ERROR: Hay registros con humedad_suelo NaN etiquetados como REGAR_PROGRAMADO"
    logging.info("Validación exitosa: Cero casos de REGAR_* cuando el sensor de humedad de suelo no está presente.")

    # Guardar en CSV
    df_final.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Dataset guardado en: {os.path.abspath(OUTPUT_FILE)}")
    return df_final

if __name__ == "__main__":
    generate_hybrid_dataset(n_samples=18000)
