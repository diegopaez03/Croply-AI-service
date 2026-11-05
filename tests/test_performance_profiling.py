"""
tests/test_performance_profiling.py
-----------------------------------
Suite de análisis de rendimiento, benchmarking y perfilado de latencia (Profiling)
para el microservicio Croply AI Service.

Mide con precisión de microsegundos cada una de las fases del pipeline:
1. Fase Analítica (ML):
   - Preprocesamiento de telemetría y feature engineering
   - Inferencia LightGBM multi-clase + guardarraíl físico
   - Explicabilidad local con SHAP TreeExplainer
   - Reglas agroclimáticas INTA y fallback determinista
2. Fase Generativa (LLM):
   - Construcción y contextualización del prompt
   - Llamada HTTP asíncrona a Ollama (Gemma 4)
   - Parsing JSON y validación Pydantic
"""

import sys
import os
import time
import statistics
import logging
from typing import List, Dict, Any
from fastapi.testclient import TestClient

# Asegurar importación desde la raíz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
client = TestClient(app)

# Payloads de prueba representativos
PAYLOAD_ESTRES_HIDRICO = {
    "parcela_id": "benchmark-tomate-estres",
    "cultivo": "tomate",
    "dias_desde_siembra": 60,
    "etapa_fenologica_idx": 2,
    "telemetria_iot": {
        "temp_ambiente_c": 28.5,
        "humedad_ambiente_pct": 38.0,
        "humedad_suelo_pct": 14.8,
        "radiacion_solar_wm2": 820.0,
        "ph_suelo": 6.4,
        "lluvia_mm_24h": 0.0
    },
    "pronostico_meteo": {
        "temp_min_pronostico": 15.0,
        "temp_max_pronostico": 32.0,
        "precip_pronostico_24h": 0.0,
        "viento_kmh": 12.0,
        "wind_gust_kmh": 16.0,
        "et0_diario": 5.8
    },
    "historial_manejo": {
        "dias_desde_ultimo_riego": 5,
        "dias_desde_ultima_curacion": 12
    }
}

PAYLOAD_ESTADO_OPTIMO = {
    "parcela_id": "benchmark-lechuga-optima",
    "cultivo": "lechuga",
    "dias_desde_siembra": 25,
    "etapa_fenologica_idx": 1,
    "telemetria_iot": {
        "temp_ambiente_c": 19.5,
        "humedad_ambiente_pct": 65.0,
        "humedad_suelo_pct": 27.5,
        "radiacion_solar_wm2": 550.0,
        "ph_suelo": 6.5,
        "lluvia_mm_24h": 0.0
    },
    "pronostico_meteo": {
        "temp_min_pronostico": 10.0,
        "temp_max_pronostico": 24.0,
        "precip_pronostico_24h": 0.0,
        "viento_kmh": 6.0,
        "wind_gust_kmh": 8.0
    },
    "historial_manejo": {
        "dias_desde_ultimo_riego": 2,
        "dias_desde_ultima_curacion": 3
    }
}


def benchmark_fase_analitica_ml(iteraciones: int = 20) -> Dict[str, Any]:
    """Mide la distribución estadística de la fase analítica (LightGBM + SHAP)."""
    print("\n" + "=" * 78)
    print(f"  BENCHMARK: FASE ANALÍTICA ML (LightGBM + SHAP) - {iteraciones} ITERACIONES")
    print("=" * 78)
    
    # Warmup
    _ = client.post("/api/v1/ai/evaluate-parcel", json=PAYLOAD_ESTADO_OPTIMO)
    
    t_totales = []
    t_preps = []
    t_lgbms = []
    t_shaps = []
    t_reglas = []
    
    for i in range(iteraciones):
        resp = client.post("/api/v1/ai/evaluate-parcel", json=PAYLOAD_ESTADO_OPTIMO)
        data = resp.json()
        met = data["metricas_rendimiento"]["fase_analitica_ml"]
        t_totales.append(met["latencia_analyzer_total_ms"])
        t_preps.append(met["preprocesamiento_features_ms"])
        t_lgbms.append(met["inferencia_lightgbm_ms"])
        t_shaps.append(met["explicabilidad_shap_ms"])
        t_reglas.append(met["reglas_climaticas_fallback_ms"])

    def calc_stats(arr):
        return {
            "media": round(statistics.mean(arr), 2),
            "mediana": round(statistics.median(arr), 2),
            "min": round(min(arr), 2),
            "max": round(max(arr), 2),
            "p95": round(statistics.quantiles(arr, n=20)[18], 2) if len(arr) >= 20 else round(max(arr), 2)
        }

    stats = {
        "Total Analizador ML": calc_stats(t_totales),
        "1. Preprocesamiento & Features": calc_stats(t_preps),
        "2. Inferencia LightGBM + Máscara": calc_stats(t_lgbms),
        "3. Explicabilidad Local SHAP": calc_stats(t_shaps),
        "4. Reglas Climáticas & Fallback": calc_stats(t_reglas),
    }

    print(f"{'Componente / Fase':<34} | {'Media':>8} | {'Mediana':>8} | {'Mín':>8} | {'Máx':>8} | {'P95':>8}")
    print("-" * 78)
    for nombre, s in stats.items():
        print(f"{nombre:<34} | {s['media']:>7.2f}ms | {s['mediana']:>7.2f}ms | {s['min']:>7.2f}ms | {s['max']:>7.2f}ms | {s['p95']:>7.2f}ms")
    print("-" * 78)
    return stats


def test_perfil_flujo_estado_optimo():
    """Valida el perfil de latencia en Estado Óptimo (HU-NA-03: sin LLM)."""
    print("\n" + "=" * 78)
    print("  TEST DE RENDIMIENTO: ESTADO ÓPTIMO (HU-NA-03 - LLM OMITIDO)")
    print("=" * 78)
    
    t0 = time.perf_counter()
    resp = client.post("/api/v1/ai/evaluate-parcel", json=PAYLOAD_ESTADO_OPTIMO)
    elapsed_external = (time.perf_counter() - t0) * 1000
    
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    met = data["metricas_rendimiento"]
    ml = met["fase_analitica_ml"]
    llm = met["fase_generativa_llm"]
    
    print(f"Latencia Total Servidor : {met['latencia_total_ms']:.2f} ms")
    print(f"Latencia Cliente E2E    : {elapsed_external:.2f} ms")
    print(f"Acción Predicha         : {data['accion_recomendada']}")
    print(f"Requiere Intervención   : {data['requiere_intervencion_llm']} (LLM Invocado: {llm['invoco_llm']})")
    print("\nDesglose Detallado:")
    print(f"  • Preprocesamiento Features : {ml['preprocesamiento_features_ms']:>6.2f} ms")
    print(f"  • Inferencia LightGBM       : {ml['inferencia_lightgbm_ms']:>6.2f} ms")
    print(f"  • Explicabilidad SHAP       : {ml['explicabilidad_shap_ms']:>6.2f} ms")
    print(f"  • Reglas Climáticas INTA    : {ml['reglas_climaticas_fallback_ms']:>6.2f} ms")
    print(f"  • Orquestador LLM (Omitido) : {llm['latencia_recommender_total_ms']:>6.2f} ms ({llm['pct_del_total']:.1f}%)")
    print("-" * 78)
    
    # Criterio: En estado óptimo debe resolverse en menos de 60 ms
    assert met["latencia_total_ms"] < 60.0, f"Latencia alta en estado óptimo: {met['latencia_total_ms']} ms"
    assert llm["invoco_llm"] is False


def test_perfil_flujo_estres_hidrico():
    """Valida el perfil de latencia completo cuando se dispara recomendación."""
    print("\n" + "=" * 78)
    print("  TEST DE RENDIMIENTO: RECOMENDACIÓN ACTIVA (ESTRÉS HÍDRICO - TOMATE)")
    print("=" * 78)
    
    t0 = time.perf_counter()
    resp = client.post("/api/v1/ai/evaluate-parcel", json=PAYLOAD_ESTRES_HIDRICO)
    elapsed_external = (time.perf_counter() - t0) * 1000
    
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    met = data["metricas_rendimiento"]
    ml = met["fase_analitica_ml"]
    llm = met["fase_generativa_llm"]
    
    print(f"Latencia Total Servidor : {met['latencia_total_ms']:.2f} ms ({met['latencia_total_ms']/1000:.2f} s)")
    print(f"Latencia Cliente E2E    : {elapsed_external:.2f} ms")
    print(f"Acción Predicha         : {data['accion_recomendada']}")
    print(f"Fuente de Generación    : {data['fuente_generacion']}")
    print(f"Título Corto            : {data['recomendacion_vigente']['titulo_corto']}")
    print(f"Texto Breve             : {data['recomendacion_vigente']['texto_breve']}")
    print("\nDesglose Detallado:")
    print(f"  [1] Fase Analítica ML (LightGBM + SHAP): {ml['latencia_analyzer_total_ms']:>7.2f} ms ({ml['pct_del_total']:>5.2f}%)")
    print(f"      - Preprocesamiento & Features      : {ml['preprocesamiento_features_ms']:>7.2f} ms")
    print(f"      - Inferencia LightGBM + Guardarraíl : {ml['inferencia_lightgbm_ms']:>7.2f} ms")
    print(f"      - Explicabilidad Local SHAP        : {ml['explicabilidad_shap_ms']:>7.2f} ms")
    print(f"      - Reglas Climáticas & Fallback     : {ml['reglas_climaticas_fallback_ms']:>7.2f} ms")
    print(f"  [2] Fase Generativa (Orquestador / LLM): {llm['latencia_recommender_total_ms']:>7.2f} ms ({llm['pct_del_total']:>5.2f}%)")
    print(f"      - Construcción de Prompt           : {llm['tiempo_prompt_ms']:>7.2f} ms")
    print(f"      - Llamada HTTP Ollama (Gemma 4)    : {llm['tiempo_ollama_generate_ms']:>7.2f} ms")
    print(f"      - Parsing & Validación Pydantic    : {llm['tiempo_parsing_validacion_ms']:>7.2f} ms")
    print("-" * 78)


if __name__ == "__main__":
    benchmark_fase_analitica_ml(iteraciones=20)
    test_perfil_flujo_estado_optimo()
    test_perfil_flujo_estres_hidrico()
