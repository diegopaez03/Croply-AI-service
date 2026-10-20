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
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

# Cargar automáticamente variables de entorno desde archivo .env si está disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


import re

def clean_and_parse_llm_json(raw_text: str) -> dict:
    """Extrae y parsea el objeto JSON de la respuesta del modelo, tolerando markdown o bloques thinking."""
    text = raw_text.strip()
    # 1. Remover bloques de razonamiento si estuvieran presentes
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 2. Si viene envuelto en markdown ```json ... ```
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
            
    # 3. Intentar parse directo
    try:
        return json.loads(text)
    except Exception:
        pass
        
    # 4. Buscar el primer '{' y el último '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return json.loads(text[first_brace:last_brace + 1])
        
    raise ValueError(f"No se pudo extraer un JSON válido de la respuesta de Ollama: {raw_text[:200]}")


def get_ollama_url() -> str:
    """Obtiene la URL de Ollama soportando OLLAMA_BASE_URL y OLLAMA_URL."""
    return os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434"


def get_ollama_model() -> str:
    """Obtiene el identificador del modelo Ollama configurado (default: gemma4:e4b)."""
    return os.getenv("OLLAMA_MODEL") or "gemma4:e4b"


def get_ollama_timeout() -> float:
    """Obtiene el timeout en segundos para la llamada al LLM."""
    return float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "75.0"))


OLLAMA_URL = get_ollama_url()
OLLAMA_MODEL = get_ollama_model()
TIMEOUT_SECONDS = get_ollama_timeout()


class RecommendationTextSchema(BaseModel):
    """Esquema estricto de salida para la entidad RecomendacionIA (HU-NA-03)."""
    titulo_corto: str = Field(..., max_length=40, description="Título imperativo o badge de acción (<= 40 caracteres)")
    texto_breve: str = Field(..., max_length=120, description="Resumen directo con datos del sensor principal (<= 120 caracteres)")
    texto_explicativo_extendido: str = Field(..., description="Explicación agronómica detallada y empática de 2 a 3 oraciones")

    @field_validator("titulo_corto", mode="before")
    @classmethod
    def clamp_titulo(cls, v: Any) -> str:
        s = str(v).strip()
        if len(s) > 40:
            return s[:37] + "..."
        return s

    @field_validator("texto_breve", mode="before")
    @classmethod
    def clamp_texto_breve(cls, v: Any) -> str:
        s = str(v).strip()
        if len(s) > 120:
            return s[:117] + "..."
        return s


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
    """Construye un prompt conciso y directo para Gemma 4 en Ollama."""
    meta = analyzer_payload["metadata"]
    pred = analyzer_payload["prediccion_ml"]
    tele = analyzer_payload["telemetria_actual"]
    alertas = analyzer_payload["alertas_climaticas_activas"]
    top_factores = pred.get("explicabilidad_top3_factores", [])
    
    factores_str = ", ".join([
        f"{f['sensor_ui']}={f['valor']}"
        for f in top_factores
    ])
    
    h_suelo_str = fmt_sensor_val(tele.get('humedad_suelo_pct'), '%')
    t_amb_str = fmt_sensor_val(tele.get('temp_ambiente_c'), '°C')
    viento_str = fmt_sensor_val(tele.get('viento_kmh'), ' km/h')
    lluvia_str = fmt_sensor_val(tele.get('precip_pronostico_24h'), ' mm')
    
    return f"""Instrucción: Genera directamente el JSON para la recomendación agronómica de Croply (Cuyo, INTA) sin incluir razonamiento ni texto fuera del JSON.

Cultivo: {meta['cultivo'].upper()} (Día {meta['dias_desde_siembra']}, etapa: {meta['etapa_fenologica']})
Acción prioritaria: {pred['accion_recomendada']} (Severidad: {pred['severidad']}, Confianza: {pred['confianza'] * 100:.1f}%)
Sensor determinante: {pred['sensor_origen_principal']}
Factores SHAP: {factores_str}
Alertas climáticas: {alertas if alertas else 'Ninguna'}
Telemetría: Suelo={h_suelo_str}, Temp={t_amb_str}, Viento={viento_str}, Lluvia 24h={lluvia_str}.

Esquema JSON requerido:
{{
  "titulo_corto": "Frase imperativa directa <= 40 caracteres",
  "texto_breve": "Síntesis con datos del sensor principal <= 120 caracteres",
  "texto_explicativo_extendido": "2 a 3 oraciones empáticas explicando la causa agronómica y recomendación."
}}
"""


async def generate_recommendation(analyzer_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquesta la generación de recomendaciones en base al payload en memoria de analyzer.py.
    Mide granularmente cada fase (construcción de prompt, llamada HTTP a Ollama y validación Pydantic).
    """
    t_rec_start = time.perf_counter()
    pred = analyzer_payload["prediccion_ml"]
    requiere_llm = pred.get("requiere_intervencion_llm", True)
    fallback = analyzer_payload.get("fallback_determinista", {})
    alertas = analyzer_payload.get("alertas_climaticas_detalle", [])
    
    # -------------------------------------------------------------
    # 1. Regla de Estado Vacío (HU-NA-03)
    # -------------------------------------------------------------
    if not requiere_llm:
        logging.info("[recommender.py] Parcela en estado óptimo sin alertas. Aplicando regla de Estado Vacío (HU-NA-03).")
        t_rec_ms = (time.perf_counter() - t_rec_start) * 1000
        return {
            "recomendacion_vigente": None,
            "mensaje_ui": "No hay recomendaciones activas en este momento.",
            "fuente_generacion": "estado_optimo_sin_accion",
            "accion_recomendada": pred["accion_recomendada"],
            "severidad": pred["severidad"],
            "sensor_origen_principal": pred["sensor_origen_principal"],
            "alertas_climaticas": alertas,
            "desglose_latencia_recommender": {
                "latencia_recommender_ms": round(t_rec_ms, 3),
                "invoco_llm": False,
                "tiempo_prompt_ms": 0.0,
                "tiempo_ollama_generate_ms": 0.0,
                "tiempo_parsing_validacion_ms": 0.0
            }
        }

    # -------------------------------------------------------------
    # 2. Invocación Asíncrona a Ollama con Gemma 4
    # -------------------------------------------------------------
    t_prompt_start = time.perf_counter()
    prompt = build_gemma_prompt(analyzer_payload)
    t_prompt_ms = (time.perf_counter() - t_prompt_start) * 1000
    
    current_url = get_ollama_url()
    current_model = get_ollama_model()
    current_timeout = get_ollama_timeout()
    
    t_http_ms = 0.0
    t_parse_ms = 0.0
    
    try:
        logging.info(f"[recommender.py] Solicitando redacción a Ollama ({current_model}) en {current_url}...")
        t_http_start = time.perf_counter()
        async with httpx.AsyncClient(timeout=current_timeout) as client:
            resp = await client.post(
                f"{current_url}/api/generate",
                json={
                    "model": current_model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 800
                    }
                }
            )
        t_http_ms = (time.perf_counter() - t_http_start) * 1000
            
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama respondió con código de estado HTTP {resp.status_code}: {resp.text}")
            
        t_parse_start = time.perf_counter()
        resp_data = resp.json()
        raw_response = resp_data.get("response", "{}")
        parsed_json = clean_and_parse_llm_json(raw_response)
        
        # Validación de longitud y tipos con Pydantic
        validated_rec = RecommendationTextSchema(**parsed_json)
        t_parse_ms = (time.perf_counter() - t_parse_start) * 1000
        
        t_rec_total_ms = (time.perf_counter() - t_rec_start) * 1000
        logging.info(f"[recommender.py] Recomendación generada exitosamente por Gemma 4 en {t_rec_total_ms:.1f} ms.")
        return {
            "recomendacion_vigente": validated_rec.model_dump(),
            "mensaje_ui": validated_rec.texto_breve,
            "fuente_generacion": "ollama_gemma4",
            "modelo_utilizado": current_model,
            "accion_recomendada": pred["accion_recomendada"],
            "severidad": pred["severidad"],
            "sensor_origen_principal": pred["sensor_origen_principal"],
            "alertas_climaticas": alertas,
            "desglose_latencia_recommender": {
                "latencia_recommender_ms": round(t_rec_total_ms, 2),
                "invoco_llm": True,
                "tiempo_prompt_ms": round(t_prompt_ms, 3),
                "tiempo_ollama_generate_ms": round(t_http_ms, 2),
                "tiempo_parsing_validacion_ms": round(t_parse_ms, 3)
            }
        }

    except (httpx.ConnectError, httpx.TimeoutException, ValidationError, json.JSONDecodeError, RuntimeError, Exception) as err:
        t_rec_total_ms = (time.perf_counter() - t_rec_start) * 1000
        if "t_http_start" in locals() and t_http_ms == 0.0:
            t_http_ms = (time.perf_counter() - t_http_start) * 1000
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
            "alertas_climaticas": alertas,
            "desglose_latencia_recommender": {
                "latencia_recommender_ms": round(t_rec_total_ms, 2),
                "invoco_llm": True,
                "tiempo_prompt_ms": round(t_prompt_ms, 3),
                "tiempo_ollama_generate_ms": round(t_http_ms, 2),
                "tiempo_parsing_validacion_ms": round(t_parse_ms, 3),
                "error": f"{type(err).__name__}: {str(err)}"
            }
        }
