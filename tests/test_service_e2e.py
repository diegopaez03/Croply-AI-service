"""
Croply - Test de Integración End-to-End (test_service_e2e.py)
============================================================
Prueba automatizada de los 3 flujos críticos del servicio FastAPI:
- Flujo A: Parcela con estrés hídrico (déficit severo en Tomate -> REGAR_URGENTE).
- Flujo B: Parcela con sensores faltantes (null) y viento fuerte -> POSTERGAR_CURACION_CLIMA.
- Flujo C: Parcela en estado óptimo (HU-NA-03) -> Estado Vacío sin invocar a Ollama.
- Health Check: Verificación de estado y diagnóstico general.
"""

import sys
import os
import logging
from fastapi.testclient import TestClient

# Asegurar que la raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
client = TestClient(app)


def test_health_endpoint():
    logging.info("--- [TEST 1/4] GET /api/v1/ai/health ---")
    resp = client.get("/api/v1/ai/health")
    assert resp.status_code == 200, f"Error status: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "THETA_CC" in data["edaphic_constants"]
    logging.info(f"Health check OK! Modelo versión: {data['version']}, Ollama reachable: {data['ollama_service']['reachable']}\n")


def test_flujo_a_estres_hidrico():
    logging.info("--- [TEST 2/4] FLUJO A: DÉFICIT HÍRICO SEVERO (TOMATE) ---")
    payload = {
        "parcela_id": "parcela-tomate-01",
        "cultivo": "tomate",
        "dias_desde_siembra": 60,
        "etapa_fenologica_idx": 2, # Floración
        "telemetria_iot": {
            "temp_ambiente_c": 28.5,
            "humedad_ambiente_pct": 38.0,
            "humedad_suelo_pct": 14.8, # Suelo severamente seco
            "radiacion_solar_wm2": 780.0,
            "ph_suelo": 6.4,
            "lluvia_mm_24h": 0.0
        },
        "pronostico_meteo": {
            "temp_min_pronostico": 15.0,
            "temp_max_pronostico": 31.0,
            "precip_pronostico_24h": 0.0,
            "viento_kmh": 8.0,
            "wind_gust_kmh": 12.0
        },
        "historial_manejo": {
            "dias_desde_ultimo_riego": 6,
            "dias_desde_ultima_curacion": 5
        }
    }
    
    resp = client.post("/api/v1/ai/evaluate-parcel", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    
    logging.info(f"Acción predicha: {data['accion_recomendada']} (Severidad: {data['severidad']}, Confianza: {data['confianza_ml']:.2%})")
    logging.info(f"Sensor origen: {data['sensor_origen_principal']}")
    logging.info(f"Fuente de recomendación: {data['fuente_generacion']}")
    logging.info(f"Título corto: {data['recomendacion_vigente']['titulo_corto']}")
    logging.info(f"Texto breve: {data['recomendacion_vigente']['texto_breve']}")
    logging.info(f"Latencia analyzer: {data['latencia_analyzer_ms']} ms | Latencia total: {data['latencia_total_ms']} ms")
    
    assert data["accion_recomendada"] in ["REGAR_URGENTE", "REGAR_PROGRAMADO"], f"Acción inesperada: {data['accion_recomendada']}"
    assert data["sensor_origen_principal"] == "humedad_suelo_pct"
    assert data["requiere_intervencion_llm"] is True
    assert data["recomendacion_vigente"] is not None
    assert len(data["recomendacion_vigente"]["titulo_corto"]) <= 40
    assert len(data["recomendacion_vigente"]["texto_breve"]) <= 120
    assert data["latencia_analyzer_ms"] < 60.0, f"Latencia del analizador alta: {data['latencia_analyzer_ms']} ms"
    logging.info("Flujo A validado exitosamente.\n")


def test_flujo_b_sensores_faltantes():
    logging.info("--- [TEST 3/5] FLUJO B: SENSORES FALTANTES TOTALES (AMBIENTE + SUELO EN NULL) + VIENTO FUERTE (AJO) ---")
    payload = {
        "parcela_id": "parcela-ajo-sin-sensor-suelo",
        "cultivo": "ajo",
        "dias_desde_siembra": 70,
        "etapa_fenologica_idx": 1,
        "telemetria_iot": {
            "temp_ambiente_c": None, # SENSOR AMBIENTAL FALTANTE (Fallback a pronóstico)
            "humedad_ambiente_pct": None, # SENSOR AMBIENTAL FALTANTE (Fallback a 50%)
            "humedad_suelo_pct": None, # SENSOR SUELO FALTANTE
            "radiacion_solar_wm2": None, # SENSOR FALTANTE
            "ph_suelo": None, # SENSOR FALTANTE
            "lluvia_mm_24h": 0.0
        },
        "pronostico_meteo": {
            "temp_min_pronostico": 11.0,
            "temp_max_pronostico": 27.0,
            "precip_pronostico_24h": 0.0,
            "viento_kmh": 26.0, # Viento alto (>15 km/h)
            "wind_gust_kmh": 34.0
        },
        "historial_manejo": {
            "dias_desde_ultimo_riego": 4,
            "dias_desde_ultima_curacion": 18 # Cura pendiente (>14 días)
        }
    }
    
    resp = client.post("/api/v1/ai/evaluate-parcel", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    
    logging.info(f"Acción predicha: {data['accion_recomendada']}")
    logging.info(f"Sensor origen: {data['sensor_origen_principal']}")
    logging.info(f"Factores SHAP Top 3 (deben excluir cultivo_enc, dias_siembra, hora):")
    for f in data["explicabilidad_top3_factores"]:
        logging.info(f"  - {f['sensor_ui']}: {f['valor']} (SHAP: {f['shap_attribution']})")
        assert f["valor"] is not None, "Error: Factor SHAP no debe ser None"
        assert f["feature"] not in ["cultivo_enc", "dias_desde_siembra", "etapa_fenologica_idx", "hora_del_dia"], f"Variable no física en Top 3: {f['feature']}"
        
    # Guardarraíl: Sin sensor de suelo ni pH, no puede recomendar regar ni corregir pH
    assert data["accion_recomendada"] not in ["REGAR_URGENTE", "REGAR_PROGRAMADO", "CORREGIR_PH_SUELO"]
    assert data["accion_recomendada"] == "POSTERGAR_CURACION_CLIMA"
    assert data["sensor_origen_principal"] == "viento_kmh"
    logging.info("Flujo B validado exitosamente.\n")


def test_flujo_c_estado_vacio():
    logging.info("--- [TEST 4/5] FLUJO C: PARCELA EN ESTADO ÓPTIMO (ESTADO VACÍO HU-NA-03) ---")
    payload = {
        "parcela_id": "parcela-lechuga-optima",
        "cultivo": "lechuga",
        "dias_desde_siembra": 25,
        "etapa_fenologica_idx": 1,
        "telemetria_iot": {
            "temp_ambiente_c": 19.5,
            "humedad_ambiente_pct": 65.0,
            "humedad_suelo_pct": 27.5, # Humedad óptima cerca de capacidad de campo (30%)
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
    
    resp = client.post("/api/v1/ai/evaluate-parcel", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    
    logging.info(f"Acción predicha: {data['accion_recomendada']}")
    logging.info(f"Requiere intervención LLM: {data['requiere_intervencion_llm']}")
    logging.info(f"Recomendación vigente: {data['recomendacion_vigente']}")
    logging.info(f"Mensaje UI: {data['mensaje_ui']}")
    logging.info(f"Fuente de generación: {data['fuente_generacion']}")
    
    assert data["accion_recomendada"] == "ESTADO_OPTIMO_SIN_ACCION"
    assert data["requiere_intervencion_llm"] is False
    assert data["recomendacion_vigente"] is None, "En estado óptimo recomendacion_vigente debe ser null"
    assert data["mensaje_ui"] == "No hay recomendaciones activas en este momento."
    assert data["fuente_generacion"] == "estado_optimo_sin_accion"
    assert len(data["alertas_climaticas"]) == 0
    logging.info("Flujo C validado exitosamente.\n")


def test_flujo_d_alerta_en_estado_optimo():
    logging.info("--- [TEST 5/5] FLUJO D: ALERTA CLIMÁTICA ACTIVA EN ESTADO ÓPTIMO ---")
    payload = {
        "parcela_id": "parcela-tomate-helada-alerta",
        "cultivo": "tomate",
        "dias_desde_siembra": 40,
        "etapa_fenologica_idx": 1,
        "telemetria_iot": {
            "temp_ambiente_c": 16.0,
            "humedad_ambiente_pct": 55.0,
            "humedad_suelo_pct": 27.0, # Suelo óptimo
            "radiacion_solar_wm2": 450.0,
            "ph_suelo": 6.5,
            "lluvia_mm_24h": 0.0
        },
        "pronostico_meteo": {
            "temp_min_pronostico": -2.0, # HELADA INMINENTE (<= 0°C)
            "temp_max_pronostico": 18.0,
            "precip_pronostico_24h": 0.0,
            "viento_kmh": 6.0,
            "wind_gust_kmh": 8.0
        },
        "historial_manejo": {
            "dias_desde_ultimo_riego": 2,
            "dias_desde_ultima_curacion": 3
        }
    }
    
    resp = client.post("/api/v1/ai/evaluate-parcel", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    
    logging.info(f"Acción predicha: {data['accion_recomendada']}")
    logging.info(f"Requiere intervención LLM: {data['requiere_intervencion_llm']}")
    logging.info(f"Alertas climáticas: {data['alertas_climaticas']}")
    logging.info(f"Recomendación vigente: {data['recomendacion_vigente']}")
    
    # Debe predecir estado óptimo de suelo, pero requerir intervención por helada
    assert data["accion_recomendada"] == "ESTADO_OPTIMO_SIN_ACCION"
    assert data["requiere_intervencion_llm"] is True
    assert len(data["alertas_climaticas"]) > 0
    assert any(a["codigo_alerta"] == "ALERTA_HELADA" for a in data["alertas_climaticas"])
    assert data["recomendacion_vigente"] is not None
    # El título o texto debe advertir sobre la alerta
    rec = data["recomendacion_vigente"]
    assert "Helada" in rec["titulo_corto"] or "Precaución" in rec["titulo_corto"] or "helada" in rec["texto_breve"].lower()
    logging.info("Flujo D validado exitosamente.\n")


def run_all_tests():
    logging.info("======================================================================")
    logging.info("EJECUTANDO SUITE DE INTEGRACIÓN E2E CROPLY AI (FASTAPI)")
    logging.info("======================================================================\n")
    test_health_endpoint()
    test_flujo_a_estres_hidrico()
    test_flujo_b_sensores_faltantes()
    test_flujo_c_estado_vacio()
    test_flujo_d_alerta_en_estado_optimo()
    logging.info("======================================================================")
    logging.info("TODOS LOS TESTS E2E PASARON EXITOSAMENTE (5/5)")
    logging.info("======================================================================\n")


if __name__ == "__main__":
    run_all_tests()
