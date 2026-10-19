"""
Croply - Orquestador de Lenguaje Natural (recommender.py)
=========================================================
Consume en memoria la salida estructurada de analyzer.py y genera la recomendación
agronómica para el productor mediante Gemma 4 en Ollama o su fallback determinista.

Criterios de producción:
1. Regla de Estado Vacío (HU-NA-03): si requiere_intervencion_llm == False,
   retorna inmediatamente sin invocar al LLM.
2. Llamada asíncrona a Ollama (POST http://localhost:11434/api/generate)
   con httpx.AsyncClient(timeout=30.0) y formato JSON estructurado.
3. Validación estricta con Pydantic de titulo_corto (<=40 car), texto_breve (<=120 car)
   y texto_explicativo_extendido.
4. Tolerancia a fallos [ERR-01]: ante timeout, desconexión o fallo de validación,
   activa de forma transparente el fallback determinista precalculado por analyzer.py.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30.0"))


class RecommendationTextSchema(BaseModel):
    """Esquema estricto de salida para la entidad RecomendacionIA (HU-NA-03)."""
    titulo_corto: str = Field(..., max_length=40, description="Título imperativo o badge de acción (<= 40 caracteres)")
    texto_breve: str = Field(..., max_length=120, description="Resumen directo con datos del sensor principal (<= 120 caracteres)")
    texto_explicativo_extendido: str = Field(..., description="Explicación agronómica detallada y empática de 2 a 3 oraciones")


import math

def fmt_sensor_val(val: Any, unit: str = "") -> str:
    """Formatea valores de sensores para evitar 'None%' en prompts a LLMs."""
    if val is None:
        return "sin sensor / sin señal"
    try:
        if math.isnan(float(val)):
            return "sin sensor / sin señal"
    except (ValueError, TypeError):
        pass
    return f"{val}{unit}"


def build_gemma_prompt(analyzer_payload: Dict[str, Any]) -> str:
    """Construye un prompt de alta precisión para Gemma 4 en Ollama."""
    meta = analyzer_payload["metadata"]
    pred = analyzer_payload["prediccion_ml"]
    tele = analyzer_payload["telemetria_actual"]
    alertas = analyzer_payload["alertas_climaticas_activas"]
    top_factores = pred.get("explicabilidad_top3_factores", [])
    
    factores_str = ", ".join([
        f"{f['sensor_ui']} ({f['valor']}) con atribución {f['shap_attribution']}"
        for f in top_factores
    ])
    
    h_suelo_str = fmt_sensor_val(tele.get('humedad_suelo_pct'), '%')
    t_amb_str = fmt_sensor_val(tele.get('temp_ambiente_c'), '°C')
    viento_str = fmt_sensor_val(tele.get('viento_kmh'), ' km/h')
    lluvia_str = fmt_sensor_val(tele.get('precip_pronostico_24h'), ' mm')
    
    prompt = f"""Eres el Asistente Agronómico Experto de Croply, especializado en horticultura y vitivinicultura de Cuyo (Mendoza, Argentina) bajo pautas del INTA.

CONTEXTO DE LA PARCELA:
- Cultivo: {meta['cultivo'].upper()} (Día {meta['dias_desde_siembra']}, etapa fenológica: {meta['etapa_fenologica']})
- Acción agronómica prioritaria obligatoria: {pred['accion_recomendada']}
- Nivel de Severidad: {pred['severidad']} (Confianza del modelo: {pred['confianza'] * 100:.1f}%)
- Sensor físico determinante: {pred['sensor_origen_principal']}
- Factores clave detectados (SHAP): {factores_str}
- Alertas climáticas activas: {alertas if alertas else 'Ninguna'}
- Telemetría actual: Humedad suelo={h_suelo_str}, Temp={t_amb_str}, Viento={viento_str}, Pronóstico lluvia 24h={lluvia_str}.

REGLAS INVIOLABLES DE RESPUESTA:
1. Responde ÚNICAMENTE un objeto JSON válido con las claves "titulo_corto", "texto_breve" y "texto_explicativo_extendido".
2. "titulo_corto": Frase imperativa directa. Máximo 40 caracteres. Ejemplo: "Riego urgente recomendado".
3. "texto_breve": Síntesis con datos del sensor principal. Máximo 120 caracteres para tarjetas de la app.
4. "texto_explicativo_extendido": 2 a 3 oraciones empáticas y profesionales explicando el motivo físico agronómico y el paso a seguir, citando el sensor principal.
5. NO inventes datos que no figuren arriba. NO contradigas la acción {pred['accion_recomendada']}.

Responde exclusivamente con el JSON:"""
    return prompt


async def generate_recommendation(analyzer_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquesta la generación de recomendaciones en base al payload en memoria de analyzer.py.
    """
    pred = analyzer_payload["prediccion_ml"]
    requiere_llm = pred.get("requiere_intervencion_llm", True)
    fallback = analyzer_payload.get("fallback_determinista", {})
    alertas = analyzer_payload.get("alertas_climaticas_detalle", [])
    
    # -------------------------------------------------------------
    # 1. Regla de Estado Vacío (HU-NA-03)
    # -------------------------------------------------------------
    if not requiere_llm:
        logging.info("[recommender.py] Parcela en estado óptimo sin alertas. Aplicando regla de Estado Vacío (HU-NA-03).")
        return {
            "recomendacion_vigente": None,
            "mensaje_ui": "No hay recomendaciones activas en este momento.",
            "fuente_generacion": "estado_optimo_sin_accion",
            "accion_recomendada": pred["accion_recomendada"],
            "severidad": pred["severidad"],
            "sensor_origen_principal": pred["sensor_origen_principal"],
            "alertas_climaticas": alertas
        }

    # -------------------------------------------------------------
    # 2. Invocación Asíncrona a Ollama con Gemma 4
    # -------------------------------------------------------------
    prompt = build_gemma_prompt(analyzer_payload)
    
    try:
        logging.info(f"[recommender.py] Solicitando redacción a Ollama ({OLLAMA_MODEL}) en {OLLAMA_URL}...")
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 250
                    }
                }
            )
            
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama respondió con código de estado HTTP {resp.status_code}: {resp.text}")
            
        resp_data = resp.json()
        raw_response = resp_data.get("response", "{}")
        parsed_json = json.loads(raw_response)
        
        # Validación de longitud y tipos con Pydantic
        validated_rec = RecommendationTextSchema(**parsed_json)
        
        logging.info("[recommender.py] Recomendación generada exitosamente por Gemma 4.")
        return {
            "recomendacion_vigente": validated_rec.model_dump(),
            "mensaje_ui": validated_rec.texto_breve,
            "fuente_generacion": "ollama_gemma4",
            "modelo_utilizado": OLLAMA_MODEL,
            "accion_recomendada": pred["accion_recomendada"],
            "severidad": pred["severidad"],
            "sensor_origen_principal": pred["sensor_origen_principal"],
            "alertas_climaticas": alertas
        }

    except (httpx.ConnectError, httpx.TimeoutException, ValidationError, json.JSONDecodeError, RuntimeError, Exception) as err:
        logging.warning(f"[recommender.py] [ERR-01] Error o timeout en servicio Ollama Gemma 4 ({type(err).__name__}: {err}). "
                        f"Activando fallback determinista de INTA.")
        
        # Validar fallback determinista con Pydantic para garantizar cumplimiento de schema
        try:
            val_fallback = RecommendationTextSchema(**fallback)
            fallback_dict = val_fallback.model_dump()
        except ValidationError:
            # Recorte de seguridad si excede caracteres
            fallback_dict = {
                "titulo_corto": str(fallback.get("titulo_corto", "Recomendación activa"))[:40],
                "texto_breve": str(fallback.get("texto_breve", "Atención requerida en la parcela."))[:120],
                "texto_explicativo_extendido": str(fallback.get("texto_explicativo_extendido", "Se detectaron condiciones que requieren intervención."))
            }
            
        return {
            "recomendacion_vigente": fallback_dict,
            "mensaje_ui": fallback_dict["texto_breve"],
            "fuente_generacion": "fallback_determinista_inta",
            "motivo_fallback": f"[ERR-01] {type(err).__name__}",
            "accion_recomendada": pred["accion_recomendada"],
            "severidad": pred["severidad"],
            "sensor_origen_principal": pred["sensor_origen_principal"],
            "alertas_climaticas": alertas
        }
