"""Bypass de redacción y evaluación climática sin LLM."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analyzer import CroplyAnalyzer
from app.main import (
    EvaluateClimateRequest,
    EvaluateParcelRequest,
    evaluate_climate,
    evaluate_parcel,
)


def _analisis(accion: str) -> dict:
    return {
        "metadata": {
            "cultivo": "tomate",
            "etapa_fenologica": "media",
            "latencia_analyzer_ms": 1.0,
            "timestamp": "2026-10-23T12:00:00Z",
            "desglose_latencia_analyzer": {},
        },
        "prediccion_ml": {
            "accion_recomendada": accion,
            "severidad": "Alta",
            "confianza": 0.9,
            "sensor_origen_principal": "humedad_suelo_pct",
            "requiere_intervencion_llm": True,
            "explicabilidad_top3_factores": [
                {"sensor_ui": "humedad_suelo_pct", "valor": 14.8, "feature": "humedad_suelo_pct", "shap_attribution": 1.0}
            ],
        },
        "alertas_climaticas_detalle": [],
        "riesgo_climatico_fenologico": {"nivel": "BAJO"},
        "telemetria_actual": {},
        "fallback_determinista": {
            "titulo_corto": "Regar",
            "texto_breve": "Riego",
            "texto_explicativo_extendido": "Regar el lote.",
        },
    }


def _pedido(accion_previa: str | None = None) -> EvaluateParcelRequest:
    datos = {
        "parcela_id": "12",
        "cultivo": "tomate",
        "dias_desde_siembra": 40,
        "telemetria_iot": {"humedad_suelo_pct": 14.8},
        "pronostico_meteo": {
            "temp_min_pronostico": 10.0,
            "temp_max_pronostico": 28.0,
            "precip_pronostico_24h": 0.0,
            "viento_kmh": 8.0,
        },
        "historial_manejo": {
            "dias_desde_ultimo_riego": 3,
            "dias_desde_ultima_curacion": 10,
        },
    }
    if accion_previa is not None:
        datos["accion_previa"] = accion_previa
    return EvaluateParcelRequest(**datos)


def _analyzer_clima() -> CroplyAnalyzer:
    return object.__new__(CroplyAnalyzer)


def test_misma_accion_no_invoca_llm():
    analyzer = _analyzer_clima()
    analyzer.analyze_parcel = lambda _payload: _analisis("REGAR_URGENTE")
    generador = AsyncMock(side_effect=AssertionError("no debe redactar"))

    async def correr():
        with patch("app.main.get_analyzer", return_value=analyzer), patch(
            "app.main.generate_recommendation", generador
        ):
            return await evaluate_parcel(_pedido("REGAR_URGENTE"))

    data = asyncio.run(correr())
    generador.assert_not_called()
    assert data["fuente_generacion"] == "bypass_sin_llm"
    assert data["recomendacion_vigente"] is None
    assert data["accion_recomendada"] == "REGAR_URGENTE"
    assert data["requiere_intervencion_llm"] is False


def test_accion_distinta_si_invoca_llm():
    analyzer = _analyzer_clima()
    analyzer.analyze_parcel = lambda _payload: _analisis("SUSPENDER_RIEGO")
    generador = AsyncMock(
        return_value={
            "recomendacion_vigente": {
                "titulo_corto": "Suspender riego",
                "texto_breve": "Hay agua de sobra.",
                "texto_explicativo_extendido": "No regar.",
            },
            "mensaje_ui": "Hay agua de sobra.",
            "fuente_generacion": "ollama_gemma4",
            "desglose_latencia_recommender": {
                "latencia_recommender_ms": 2.0,
                "invoco_llm": True,
                "proveedor_activo": "ollama",
                "tiempo_prompt_ms": 0.1,
                "tiempo_llm_generate_ms": 1.0,
                "tiempo_ollama_generate_ms": 1.0,
                "tiempo_parsing_validacion_ms": 0.1,
            },
        }
    )

    async def correr():
        with patch("app.main.get_analyzer", return_value=analyzer), patch(
            "app.main.generate_recommendation", generador
        ):
            return await evaluate_parcel(_pedido("REGAR_URGENTE"))

    data = asyncio.run(correr())
    generador.assert_awaited_once()
    assert data["fuente_generacion"] == "ollama_gemma4"
    assert data["recomendacion_vigente"]["titulo_corto"] == "Suspender riego"


def test_clima_helada_sin_llm():
    analyzer = _analyzer_clima()
    generador = AsyncMock(side_effect=AssertionError("el clima no redacta"))

    async def correr():
        with patch("app.main.get_analyzer", return_value=analyzer), patch(
            "app.main.generate_recommendation", generador
        ):
            return await evaluate_climate(
                EvaluateClimateRequest(
                    temp_min_pronostico=-2.0,
                    temp_max_pronostico=12.0,
                    precip_pronostico_24h=0.0,
                    viento_kmh=10.0,
                    humedad_relativa=45.0,
                    temp_actual=4.0,
                    contextos=[],
                )
            )

    data = asyncio.run(correr())
    generador.assert_not_called()
    assert any(a["codigo_alerta"] == "ALERTA_HELADA" for a in data["alertas_climaticas"])


def test_clima_sin_cultivo_no_dispara_helada_agronomica():
    analyzer = _analyzer_clima()
    alertas = analyzer.unir_alertas_climaticas(
        1.0, 20.0, 10.0, 12.0, 0.0, 50.0, 15.0, []
    )
    assert all(a["codigo_alerta"] != "ALERTA_HELADA_AGRONOMICA" for a in alertas)


def test_clima_une_codigos_de_varios_cultivos():
    analyzer = _analyzer_clima()
    alertas = analyzer.unir_alertas_climaticas(
        1.0,
        20.0,
        10.0,
        12.0,
        0.0,
        50.0,
        15.0,
        [
            {"cultivo": "pimiento", "etapa_fenologica_idx": 0},
            {"cultivo": "tomate", "etapa_fenologica_idx": 1},
        ],
    )
    codigos = [a["codigo_alerta"] for a in alertas]
    assert codigos.count("ALERTA_HELADA_AGRONOMICA") == 1


if __name__ == "__main__":
    test_misma_accion_no_invoca_llm()
    test_accion_distinta_si_invoca_llm()
    test_clima_helada_sin_llm()
    test_clima_sin_cultivo_no_dispara_helada_agronomica()
    test_clima_une_codigos_de_varios_cultivos()
    print("bypass y clima: ok")
