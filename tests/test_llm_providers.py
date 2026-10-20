"""
Croply - Test de Integración de Proveedores LLM (test_llm_providers.py)
======================================================================
Valida la arquitectura desacoplada de generación en lenguaje natural:
- Proveedor Ollama (Local)
- Proveedor Google Gemini API (Nube)
- Factoría de Proveedores y Conmutación Dinámica
- Guardarraíles de Validación Pydantic y Fallback Determinista ante Errores
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any
from fastapi.testclient import TestClient

# Asegurar que la raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm import (
    BaseLLMProvider,
    OllamaProvider,
    GeminiProvider,
    get_llm_provider
)
from app.recommender import (
    RecommendationTextSchema,
    generate_recommendation,
    clean_and_parse_llm_json
)
from app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
client = TestClient(app)


def test_ollama_provider_instantiation_and_health():
    logging.info("--- [TEST 1/5] OllamaProvider Instanciación y Diagnóstico ---")
    provider = OllamaProvider()
    assert provider.provider_name == "ollama"
    assert provider.model_name is not None
    
    # Diagnóstico asíncrono
    status = asyncio.run(provider.health_check())
    assert status["provider"] == "ollama"
    assert "reachable" in status
    logging.info(f"Ollama health_check OK: {status}")


def test_gemini_provider_instantiation_and_health():
    logging.info("--- [TEST 2/5] GeminiProvider Instanciación y Diagnóstico con Google GenAI ---")
    provider = GeminiProvider()
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-3-flash-preview"
    
    status = asyncio.run(provider.health_check())
    assert status["provider"] == "gemini"
    assert status["api_key_configured"] is True
    logging.info(f"Gemini health_check status: reachable={status['reachable']}, model={status['model']}")
    if status["reachable"]:
        logging.info(f"Modelo confirmado: {status.get('display_name')} (Input token limit: {status.get('input_token_limit')})")
    assert status["reachable"] is True, f"Error conectando con Gemini API: {status.get('note')}"


def test_gemini_provider_live_generation():
    logging.info("--- [TEST 3/5] GeminiProvider Generación Estructurada en Vivo ---")
    provider = GeminiProvider()
    prompt = """Instrucción: Genera directamente el JSON para la recomendación agronómica de Croply (Cuyo, INTA) sin incluir razonamiento ni texto fuera del JSON.

Cultivo: TOMATE (Día 60, etapa: media)
Acción prioritaria: REGAR_URGENTE (Severidad: CRITICA, Confianza: 94.0%)
Sensor determinante: humedad_suelo_pct
Factores SHAP: humedad_suelo_pct=14.5, t_max=32.0, deficit_hidrico_mad=0.75
Alertas climáticas: Ninguna
Telemetría: Suelo=14.5%, Temp=29.0°C, Viento=10 km/h, Lluvia 24h=0.0 mm.

Esquema JSON requerido:
{
  "titulo_corto": "Frase imperativa directa <= 40 caracteres",
  "texto_breve": "Síntesis con datos del sensor principal <= 120 caracteres",
  "texto_explicativo_extendido": "2 a 3 oraciones empáticas explicando la causa agronómica y recomendación."
}
"""
    result = asyncio.run(provider.generate_recommendation_text(prompt))
    assert result.http_latency_ms > 0
    logging.info(f"Respuesta Gemini recibida en {result.http_latency_ms:.1f} ms:")
    logging.info(f"Raw text: {result.raw_text}")
    
    parsed = clean_and_parse_llm_json(result.raw_text)
    validated = RecommendationTextSchema(**parsed)
    assert len(validated.titulo_corto) <= 40
    assert len(validated.texto_breve) <= 120
    assert len(validated.texto_explicativo_extendido) > 10
    logging.info(f"Validación Pydantic exitosa: {validated.model_dump()}")


def test_factory_switching_and_fallback():
    logging.info("--- [TEST 4/5] Factoría de Proveedores y Resiliencia ante Falla ---")
    # 1. Probar factoría explícita
    ollama_prov = get_llm_provider("ollama", force_refresh=True)
    assert ollama_prov.provider_name == "ollama"

    gemini_prov = get_llm_provider("gemini", force_refresh=True)
    assert gemini_prov.provider_name == "gemini"

    # 2. Probar fallback determinista forzando error en Gemini (API Key inválida)
    broken_gemini = GeminiProvider(api_key="API_KEY_INVALIDA_DE_PRUEBA")
    dummy_payload = {
        "metadata": {
            "cultivo": "tomate",
            "dias_desde_siembra": 60,
            "etapa_fenologica": "media"
        },
        "prediccion_ml": {
            "accion_recomendada": "REGAR_URGENTE",
            "severidad": "CRITICA",
            "confianza": 0.95,
            "sensor_origen_principal": "humedad_suelo_pct",
            "requiere_intervencion_llm": True,
            "explicabilidad_top3_factores": []
        },
        "telemetria_actual": {
            "humedad_suelo_pct": 14.0,
            "temp_ambiente_c": 30.0,
            "viento_kmh": 10.0,
            "precip_pronostico_24h": 0.0
        },
        "alertas_climaticas_activas": [],
        "alertas_climaticas_detalle": [],
        "fallback_determinista": {
            "titulo_corto": "Riego urgente por déficit crítico",
            "texto_breve": "Humedad del suelo al 14.0%. Aplicar riego inmediatamente.",
            "texto_explicativo_extendido": "El perfil del suelo alcanzó el umbral crítico para tomate en etapa media."
        }
    }

    rec_res = asyncio.run(generate_recommendation(dummy_payload, provider=broken_gemini))
    assert rec_res["fuente_generacion"] == "fallback_determinista_inta"
    assert "[ERR-01]" in rec_res["motivo_fallback"]
    assert rec_res["recomendacion_vigente"]["titulo_corto"] == "Riego urgente por déficit crítico"
    logging.info(f"Fallback activado correctamente ante fallo del proveedor: {rec_res['motivo_fallback']}")


def test_e2e_evaluate_parcel_with_gemini():
    logging.info("--- [TEST 5/5] Endpoint POST /api/v1/ai/evaluate-parcel con Gemini Activo ---")
    os.environ["LLM_PROVIDER"] = "gemini"
    # Forzar refresco del provider
    get_llm_provider("gemini", force_refresh=True)

    payload = {
        "parcela_id": "parcela-test-gemini-01",
        "cultivo": "tomate",
        "dias_desde_siembra": 55,
        "etapa_fenologica_idx": 2,
        "telemetria_iot": {
            "temp_ambiente_c": 31.0,
            "humedad_ambiente_pct": 35.0,
            "humedad_suelo_pct": 15.0,
            "radiacion_solar_wm2": 820.0,
            "ph_suelo": 6.3,
            "lluvia_mm_24h": 0.0
        },
        "pronostico_meteo": {
            "temp_min_pronostico": 18.0,
            "temp_max_pronostico": 33.0,
            "precip_pronostico_24h": 0.0,
            "viento_kmh": 12.0,
            "wind_gust_kmh": 18.0
        },
        "historial_manejo": {
            "dias_desde_ultimo_riego": 5,
            "dias_desde_ultima_curacion": 4
        }
    }

    resp = client.post("/api/v1/ai/evaluate-parcel", json=payload)
    assert resp.status_code == 200, f"Error HTTP {resp.status_code}: {resp.text}"
    data = resp.json()

    logging.info(f"Acción: {data['accion_recomendada']}")
    logging.info(f"Fuente generación: {data['fuente_generacion']}")
    logging.info(f"Título: {data['recomendacion_vigente']['titulo_corto']}")
    logging.info(f"Texto breve: {data['recomendacion_vigente']['texto_breve']}")
    logging.info(f"Métricas rendimiento: {data['metricas_rendimiento']}")

    assert data["accion_recomendada"] in ["REGAR_URGENTE", "REGAR_PROGRAMADO"]
    assert data["recomendacion_vigente"] is not None
    assert len(data["recomendacion_vigente"]["titulo_corto"]) <= 40
    assert len(data["recomendacion_vigente"]["texto_breve"]) <= 120
    assert data["metricas_rendimiento"]["fase_generativa_llm"]["proveedor_activo"] == "gemini"
    logging.info("Test E2E con Gemini finalizado exitosamente.\n")


def run_all():
    logging.info("======================================================================")
    logging.info("EJECUTANDO TEST SUITE DE PROVEEDORES LLM (OLLAMA + GOOGLE GEMINI)")
    logging.info("======================================================================\n")
    test_ollama_provider_instantiation_and_health()
    test_gemini_provider_instantiation_and_health()
    test_gemini_provider_live_generation()
    test_factory_switching_and_fallback()
    test_e2e_evaluate_parcel_with_gemini()
    logging.info("======================================================================")
    logging.info("TODOS LOS TESTS DE PROVEEDORES LLM PASARON EXITOSAMENTE (5/5)")
    logging.info("======================================================================\n")


if __name__ == "__main__":
    run_all()
