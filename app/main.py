"""
Croply - Servidor Principal de Inteligencia Artificial (main.py)
================================================================
Servicio FastAPI que expone la API unificada de evaluación agronómica
para pequeños y medianos productores de la región de Cuyo (Mendoza).

Endpoints:
- POST /api/v1/ai/evaluate-parcel: Inferencia analítica (analyzer.py) + redacción empática (recommender.py)
- GET  /api/v1/ai/health: Diagnóstico de componentes, modelo LightGBM y servicio Ollama
"""

import time
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Asegurar que la raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cargar automáticamente variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.analyzer import get_analyzer, CroplyAnalyzer
from app.recommender import (
    generate_recommendation,
    get_ollama_url,
    get_ollama_model,
    OLLAMA_URL,
    OLLAMA_MODEL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------------------------------------------
# Modelos Pydantic de Validación de Entrada y Salida
# -------------------------------------------------------------
class TelemetriaIoT(BaseModel):
    temp_ambiente_c: Optional[float] = Field(None, description="Temperatura ambiente actual en °C (nullable)", ge=-20.0, le=60.0)
    humedad_ambiente_pct: Optional[float] = Field(None, description="Humedad relativa ambiente en % (nullable)", ge=0.0, le=100.0)
    humedad_suelo_pct: Optional[float] = Field(None, description="Humedad volumétrica del suelo en % (nullable)", ge=0.0, le=100.0)
    radiacion_solar_wm2: Optional[float] = Field(None, description="Radiación solar de onda corta en W/m² (nullable)", ge=0.0, le=1500.0)
    ph_suelo: Optional[float] = Field(None, description="pH del suelo en escala 0-14 (nullable)", ge=2.0, le=12.0)
    lluvia_mm_24h: Optional[float] = Field(0.0, description="Precipitación acumulada en pluviómetro en las últimas 24h (mm)", ge=0.0)


class PronosticoMeteo(BaseModel):
    temp_min_pronostico: float = Field(..., description="Temperatura mínima pronosticada a 24-48h (°C)")
    temp_max_pronostico: float = Field(..., description="Temperatura máxima pronosticada a 24-48h (°C)")
    precip_pronostico_24h: float = Field(..., description="Precipitación acumulada prevista para las próximas 24h (mm)", ge=0.0)
    viento_kmh: float = Field(..., description="Velocidad media de viento pronosticada (km/h)", ge=0.0)
    wind_gust_kmh: Optional[float] = Field(None, description="Ráfagas máximas de viento pronosticadas (km/h)", ge=0.0)
    et0_diario: Optional[float] = Field(5.0, description="Evapotranspiración de referencia diaria (mm/día)", ge=0.0)


class HistorialManejo(BaseModel):
    dias_desde_ultimo_riego: int = Field(..., description="Días transcurridos desde el último riego", ge=0)
    dias_desde_ultima_curacion: int = Field(..., description="Días transcurridos desde la última pulverización fitosanitaria", ge=0)


class EvaluateParcelRequest(BaseModel):
    parcela_id: str = Field(..., description="Identificador único o slug de la parcela")
    finca_id: Optional[str] = Field("finca-default", description="Identificador de la finca")
    cultivo: str = Field(..., description="Nombre del cultivo (tomate, ajo, lechuga, cebolla, pimiento, papa)")
    variedad: Optional[str] = Field(None, description="Variedad comercial sembrada")
    dias_desde_siembra: int = Field(..., description="Días desde la fecha de siembra o trasplante", ge=1)
    etapa_fenologica_idx: Optional[int] = Field(None, description="Índice de etapa fenológica (0: inicial, 1: vegetativa, 2: media, 3: maduración/cosecha)", ge=0, le=3)
    hora_del_dia: Optional[int] = Field(12, description="Hora de observación en formato 24h", ge=0, le=23)
    telemetria_iot: TelemetriaIoT
    pronostico_meteo: PronosticoMeteo
    historial_manejo: HistorialManejo


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    edaphic_constants: Dict[str, float]
    ollama_service: Dict[str, Any]


# -------------------------------------------------------------
# Ciclo de Vida de la Aplicación (Lifespan)
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("[FastAPI] Inicializando Servicio de IA Croply...")
    # Cargar Singleton en memoria al arrancar
    analyzer = get_analyzer()
    logging.info(f"[FastAPI] Modelo {analyzer.version} cargado exitosamente. Listo para atender inferencias.")
    yield
    logging.info("[FastAPI] Apagando Servicio de IA Croply.")


app = FastAPI(
    title="Croply AI Service",
    description="Microservicio de Inteligencia Artificial para recomendación agronómica y alertas agroclimáticas en Cuyo.",
    version="1.3.0",
    lifespan=lifespan
)


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/api/v1/ai/health", response_model=HealthResponse, tags=["Diagnóstico"])
async def health_check():
    """Verifica el estado del modelo LightGBM y la conectividad con Ollama."""
    analyzer = get_analyzer()
    current_url = get_ollama_url()
    current_model = get_ollama_model()
    
    ollama_status = {"url": current_url, "model": current_model, "reachable": False}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{current_url}/api/tags")
            if resp.status_code == 200:
                ollama_status["reachable"] = True
                models_list = [m["name"] for m in resp.json().get("models", [])]
                ollama_status["models_available"] = models_list
                ollama_status["model_active"] = any(current_model in m for m in models_list)
    except Exception as exc:
        ollama_status["reachable"] = False
        ollama_status["note"] = f"Ollama no accesible en {current_url}: {str(exc)}. El servicio operará automáticamente con fallback_determinista_inta."

    return {
        "status": "healthy",
        "version": analyzer.version,
        "model_loaded": analyzer.model is not None,
        "edaphic_constants": analyzer.edaphic_constants,
        "ollama_service": ollama_status
    }


@app.post("/api/v1/ai/evaluate-parcel", tags=["Evaluación Agronómica"])
async def evaluate_parcel(req: EvaluateParcelRequest):
    """
    Evalúa una parcela integralmente:
    1. analyzer.py ejecuta inferencia LightGBM, reglas de viabilidad y SHAP (<5 ms).
    2. recommender.py orquesta la recomendación con Gemma 4 en Ollama o fallback determinista (HU-NA-03).
    3. Retorna la respuesta lista para persistir en PostgreSQL y renderizar en la UI.
    """
    t_start = time.perf_counter()
    analyzer = get_analyzer()
    
    # 1. Ejecutar Análisis Agronómico
    try:
        analysis_res = analyzer.analyze_parcel(req.model_dump())
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logging.error(f"Error interno en analyzer.py: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error en analyzer: {str(e)}")

    # 2. Orquestar Recomendación en Lenguaje Natural
    try:
        recommendation_res = await generate_recommendation(analysis_res)
    except Exception as e:
        logging.error(f"Error interno en recommender.py: {e}", exc_info=True)
        # Salvaguarda de emergencia
        recommendation_res = {
            "recomendacion_vigente": analysis_res.get("fallback_determinista"),
            "mensaje_ui": "Recomendación generada por reglas agronómicas de contingencia.",
            "fuente_generacion": "fallback_determinista_inta",
            "accion_recomendada": analysis_res["prediccion_ml"]["accion_recomendada"],
            "severidad": analysis_res["prediccion_ml"]["severidad"],
            "sensor_origen_principal": analysis_res["prediccion_ml"]["sensor_origen_principal"],
            "alertas_climaticas": analysis_res.get("alertas_climaticas_detalle", []),
            "desglose_latencia_recommender": {
                "latencia_recommender_ms": 0.0,
                "invoco_llm": False,
                "tiempo_prompt_ms": 0.0,
                "tiempo_ollama_generate_ms": 0.0,
                "tiempo_parsing_validacion_ms": 0.0,
                "error": str(e)
            }
        }

    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    desglose_analyzer = analysis_res["metadata"].get("desglose_latencia_analyzer", {})
    desglose_rec = recommendation_res.get("desglose_latencia_recommender", {})
    lat_analyzer = analysis_res["metadata"]["latencia_analyzer_ms"]
    lat_rec = desglose_rec.get("latencia_recommender_ms", 0.0)

    metricas_rendimiento = {
        "latencia_total_ms": total_latency_ms,
        "fase_analitica_ml": {
            "latencia_analyzer_total_ms": lat_analyzer,
            "preprocesamiento_features_ms": desglose_analyzer.get("preprocesamiento_features_ms", 0.0),
            "inferencia_lightgbm_ms": desglose_analyzer.get("inferencia_lightgbm_ms", 0.0),
            "explicabilidad_shap_ms": desglose_analyzer.get("explicabilidad_shap_ms", 0.0),
            "reglas_climaticas_fallback_ms": desglose_analyzer.get("reglas_climaticas_fallback_ms", 0.0),
            "pct_del_total": round((lat_analyzer / total_latency_ms) * 100, 2) if total_latency_ms > 0 else 0.0
        },
        "fase_generativa_llm": {
            "latencia_recommender_total_ms": lat_rec,
            "invoco_llm": desglose_rec.get("invoco_llm", False),
            "tiempo_prompt_ms": desglose_rec.get("tiempo_prompt_ms", 0.0),
            "tiempo_ollama_generate_ms": desglose_rec.get("tiempo_ollama_generate_ms", 0.0),
            "tiempo_parsing_validacion_ms": desglose_rec.get("tiempo_parsing_validacion_ms", 0.0),
            "pct_del_total": round((lat_rec / total_latency_ms) * 100, 2) if total_latency_ms > 0 else 0.0
        }
    }

    logging.info(
        f"[PERFORMANCE] Parcela '{req.parcela_id}': Total={total_latency_ms:.2f} ms | "
        f"ML={lat_analyzer:.2f} ms (Prep={desglose_analyzer.get('preprocesamiento_features_ms', 0):.2f}ms, "
        f"LGBM={desglose_analyzer.get('inferencia_lightgbm_ms', 0):.2f}ms, "
        f"SHAP={desglose_analyzer.get('explicabilidad_shap_ms', 0):.2f}ms, "
        f"Reglas={desglose_analyzer.get('reglas_climaticas_fallback_ms', 0):.2f}ms) | "
        f"LLM={lat_rec:.2f} ms (HTTP={desglose_rec.get('tiempo_ollama_generate_ms', 0):.2f}ms)"
    )
    
    # 3. Formato Unificado de Respuesta Productiva
    return {
        "parcela_id": req.parcela_id,
        "cultivo": analysis_res["metadata"]["cultivo"],
        "etapa_fenologica": analysis_res["metadata"]["etapa_fenologica"],
        "dias_desde_siembra": req.dias_desde_siembra,
        "timestamp_evaluacion": analysis_res["metadata"]["timestamp"],
        "accion_recomendada": analysis_res["prediccion_ml"]["accion_recomendada"],
        "severidad": analysis_res["prediccion_ml"]["severidad"],
        "confianza_ml": analysis_res["prediccion_ml"]["confianza"],
        "sensor_origen_principal": analysis_res["prediccion_ml"]["sensor_origen_principal"],
        "requiere_intervencion_llm": analysis_res["prediccion_ml"]["requiere_intervencion_llm"],
        "recomendacion_vigente": recommendation_res["recomendacion_vigente"],
        "mensaje_ui": recommendation_res["mensaje_ui"],
        "fuente_generacion": recommendation_res["fuente_generacion"],
        "alertas_climaticas": analysis_res["alertas_climaticas_detalle"],
        "riesgo_climatico_fenologico": analysis_res["riesgo_climatico_fenologico"],
        "explicabilidad_top3_factores": analysis_res["prediccion_ml"]["explicabilidad_top3_factores"],
        "telemetria_evaluada": analysis_res["telemetria_actual"],
        "latencia_analyzer_ms": analysis_res["metadata"]["latencia_analyzer_ms"],
        "latencia_total_ms": total_latency_ms,
        "metricas_rendimiento": metricas_rendimiento
    }
