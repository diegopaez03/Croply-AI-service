"""
Croply - Paquete de Abstracción de Modelos de Lenguaje (app.llm)
"""

from app.llm.base import BaseLLMProvider, LLMGenerationResult
from app.llm.ollama_provider import OllamaProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.factory import get_llm_provider

__all__ = [
    "BaseLLMProvider",
    "LLMGenerationResult",
    "OllamaProvider",
    "GeminiProvider",
    "get_llm_provider",
]
