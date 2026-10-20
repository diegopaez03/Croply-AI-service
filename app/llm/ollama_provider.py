"""
Croply - Proveedor LLM Ollama Local (ollama_provider.py)
========================================================
Implementa la interfaz BaseLLMProvider para interactuar de forma asíncrona
con el servidor local de Ollama (ej. Gemma 4).
"""

import os
import time
import logging
from typing import Dict, Any
import httpx

from app.llm.base import BaseLLMProvider, LLMGenerationResult


class OllamaProvider(BaseLLMProvider):
    """Proveedor para servidores Ollama locales o en red interna."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout_seconds: float = None
    ):
        self._base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        self._model = model or os.getenv("OLLAMA_MODEL") or "gemma4:e4b"
        self._timeout_seconds = timeout_seconds or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "75.0"))

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    async def generate_recommendation_text(self, prompt: str) -> LLMGenerationResult:
        """Invoca POST /api/generate en el servidor Ollama con formato JSON estricto."""
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 800
            }
        }

        logging.info(f"[OllamaProvider] Solicitando inferencia a {self._model} en {url} (timeout: {self._timeout_seconds}s)...")
        t_start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            resp = await client.post(url, json=payload)
        t_http_ms = (time.perf_counter() - t_start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama respondió HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        raw_text = data.get("response", "{}")
        return LLMGenerationResult(
            raw_text=raw_text,
            http_latency_ms=round(t_http_ms, 2),
            metadata={
                "done": data.get("done", True),
                "total_duration": data.get("total_duration")
            }
        )

    async def health_check(self) -> Dict[str, Any]:
        """Consulta GET /api/tags para diagnosticar disponibilidad del servicio y modelo."""
        status = {
            "provider": self.provider_name,
            "url": self._base_url,
            "model": self._model,
            "reachable": False
        }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code == 200:
                    status["reachable"] = True
                    models_list = [m["name"] for m in resp.json().get("models", [])]
                    status["models_available"] = models_list
                    status["model_active"] = any(self._model in m for m in models_list)
        except Exception as exc:
            status["reachable"] = False
            status["note"] = f"Ollama no accesible en {self._base_url}: {str(exc)}"

        return status
