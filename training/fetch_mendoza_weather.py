#!/usr/bin/env python3
"""
fetch_mendoza_weather.py
------------------------
Descarga datos climáticos históricos reales para Guaymallén, Mendoza (Cuyo)
utilizando la API gratuita de Open-Meteo Archive (ERA5-Land reanalysis).

Coordenadas Guaymallén: Latitude -32.91, Longitude -68.72
Rango temporal: 2024-01-01 a 2025-12-31 (2 años completos)
Variables: temperature_2m, relative_humidity_2m, precipitation,
           wind_speed_10m, shortwave_radiation, et0_fao_evapotranspiration,
           soil_moisture_7_to_28cm

Salida: raw_datasets/clima_mendoza_openmeteo.csv
"""

import os
import sys
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clima_mendoza_openmeteo.csv")

# Parámetros para Guaymallén, Mendoza
LATITUDE = -32.91
LONGITUDE = -68.72
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "America/Argentina/Mendoza"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
    "soil_moisture_7_to_28cm"
]

def fetch_weather_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE
    }
    
    logging.info(f"Consultando Open-Meteo Archive API para Guaymallén ({LATITUDE}, {LONGITUDE})...")
    logging.info(f"Rango: {START_DATE} al {END_DATE}")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logging.error(f"Error al conectar con Open-Meteo: {e}")
        sys.exit(1)
        
    if "hourly" not in data:
        logging.error(f"Respuesta inesperada de Open-Meteo: {data}")
        sys.exit(1)
        
    hourly_data = data["hourly"]
    df = pd.DataFrame(hourly_data)
    
    # Renombrar columnas para consistencia con Croply
    column_mapping = {
        "time": "timestamp",
        "temperature_2m": "temp_ambiente_c",
        "relative_humidity_2m": "humedad_ambiente_pct",
        "precipitation": "precipitacion_mm",
        "wind_speed_10m": "viento_kmh",
        "shortwave_radiation": "radiacion_solar_wm2",
        "et0_fao_evapotranspiration": "et0_mm",
        "soil_moisture_7_to_28cm": "humedad_suelo_referencia_m3m3"
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Limpieza y conversiones básicas
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Validar integridad
    logging.info(f"Registros descargados: {len(df)} horas.")
    logging.info(f"Rango real de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
    logging.info(f"Valores nulos por columna:\n{df.isnull().sum()}")
    
    # Guardar en CSV
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Datos climáticos guardados exitosamente en: {os.path.abspath(OUTPUT_FILE)}")
    
    # Imprimir estadísticas resumen
    print("\n--- RESUMEN CLIMÁTICO GUAYMALLÉN (2024-2025) ---")
    print(df.describe().round(2).to_string())

if __name__ == "__main__":
    fetch_weather_data()
