"""
Croply - Proveedor LLM Google Gemini API (gemini_provider.py)
=============================================================
Implementa la interfaz BaseLLMProvider para interactuar de forma asíncrona
con la API de Google Gemini (Google GenAI) mediante llamadas REST nativas con httpx.
No requiere dependencias pesadas adicionales (cero conflictos gRPC/protobuf).
"""

import os
import time
import logging
from typing import Dict, Any, Optional
import httpx

from app.llm.base import BaseLLMProvider, LLMGenerationResult


class GeminiProvider(BaseLLMProvider):
    """Proveedor para la API REST de Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model = model or os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview"
        self._timeout_seconds = timeout_seconds or float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30.0"))
        self._base_api_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_cleaned_model_name(self) -> str:
        """Remueve prefijo 'models/' si viniera duplicado en la configuración."""
        if self._model.startswith("models/"):
            return self._model.replace("models/", "", 1)
        return self._model

    async def generate_recommendation_text(self, prompt: str) -> LLMGenerationResult:
        """
        Ejecuta la llamada asíncrona a la API de Gemini usando response_mime_type=application/json.
        """
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno.")

        clean_model = self._get_cleaned_model_name()
        url = f"{self._base_api_url}/models/{clean_model}:generateContent?key={self._api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 800
            }
        }

        logging.info(f"[GeminiProvider] Solicitando inferencia a Google Gemini ({clean_model}) (timeout: {self._timeout_seconds}s)...")
        t_start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            resp = await client.post(url, json=payload)
        t_http_ms = (time.perf_counter() - t_start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"Google Gemini API respondió HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"Gemini no retornó candidatos de respuesta: {data}")
            
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError(f"Gemini retornó un candidato sin partes de texto: {candidates[0]}")
            
            raw_text = parts[0].get("text", "{}")
        except (KeyError, IndexError) as err:
            raise ValueError(f"Estructura inesperada en respuesta de Gemini: {err}")

        usage = data.get("usageMetadata", {})
        return LLMGenerationResult(
            raw_text=raw_text,
            http_latency_ms=round(t_http_ms, 2),
            metadata={
                "usageMetadata": usage,
                "finishReason": candidates[0].get("finishReason")
            }
        )

    async def health_check(self) -> Dict[str, Any]:
        """Verifica conectividad y validez de la API Key consultando el endpoint de metadata del modelo."""
        clean_model = self._get_cleaned_model_name()
        status = {
            "provider": self.provider_name,
            "model": clean_model,
            "api_key_configured": bool(self._api_key),
            "reachable": False
        }

        if not self._api_key:
            status["note"] = "GEMINI_API_KEY no configurada. El servicio operará automáticamente con fallback_determinista_inta."
            return status

        url = f"{self._base_api_url}/models/{clean_model}?key={self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    model_data = resp.json()
                    status["reachable"] = True
                    status["display_name"] = model_data.get("displayName")
                    status["input_token_limit"] = model_data.get("inputTokenLimit")
                else:
                    status["reachable"] = False
                    status["note"] = f"Gemini respondió HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            status["reachable"] = False
            status["note"] = f"Fallo al conectar con Google Gemini API: {str(exc)}"

        return status
