"""
Croply - Factoría de Proveedores LLM (factory.py)
=================================================
Instancia y administra el proveedor de LLM activo según la variable de entorno
LLM_PROVIDER o parámetro explícito (Strategy / Factory Pattern).
"""

import os
import logging
from typing import Optional

from app.llm.base import BaseLLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.gemini_provider import GeminiProvider

_CACHED_PROVIDER: Optional[BaseLLMProvider] = None
_CACHED_PROVIDER_TYPE: Optional[str] = None


def get_llm_provider(
    provider_name: Optional[str] = None,
    force_refresh: bool = False
) -> BaseLLMProvider:
    """
    Retorna la instancia del proveedor LLM configurado.
    Soporta 'ollama' y 'gemini'. Por defecto lee LLM_PROVIDER (default: 'ollama').
    """
    global _CACHED_PROVIDER, _CACHED_PROVIDER_TYPE

    chosen_type = (provider_name or os.getenv("LLM_PROVIDER") or "ollama").lower().strip()

    if not force_refresh and _CACHED_PROVIDER is not None and _CACHED_PROVIDER_TYPE == chosen_type:
        return _CACHED_PROVIDER

    if chosen_type == "gemini":
        logging.info("[LLM Factory] Inicializando proveedor Google Gemini API...")
        provider = GeminiProvider()
    elif chosen_type == "ollama":
        logging.info("[LLM Factory] Inicializando proveedor Ollama local...")
        provider = OllamaProvider()
    else:
        logging.warning(
            f"[LLM Factory] Proveedor desconocido '{chosen_type}'. Empleando fallback a 'ollama'."
        )
        provider = OllamaProvider()
        chosen_type = "ollama"

    _CACHED_PROVIDER = provider
    _CACHED_PROVIDER_TYPE = chosen_type
    return provider
