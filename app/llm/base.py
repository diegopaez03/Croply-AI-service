"""
Croply - Interfaz Base para Proveedores LLM (base.py)
=====================================================
Define el contrato abstracto (Strategy Pattern) que deben cumplir todos los
proveedores de Modelos de Lenguaje (Ollama, Google Gemini, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMGenerationResult:
    """Contenedor de resultado de la generación del LLM."""
    def __init__(self, raw_text: str, http_latency_ms: float, metadata: Dict[str, Any] = None):
        self.raw_text = raw_text
        self.http_latency_ms = http_latency_ms
        self.metadata = metadata or {}


class BaseLLMProvider(ABC):
    """Contrato base para cualquier motor de generación en lenguaje natural."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nombre identificador único del proveedor (ej: 'ollama', 'gemini')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre o identificador del modelo en uso (ej: 'gemma4:e4b', 'gemini-3-flash-preview')."""
        pass

    @abstractmethod
    async def generate_recommendation_text(self, prompt: str) -> LLMGenerationResult:
        """
        Ejecuta la llamada asíncrona al servicio LLM y retorna el texto en bruto
        junto con la latencia HTTP en milisegundos.
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica la conectividad y disponibilidad del servicio del proveedor.
        Retorna un dict con las claves mínimas: 'reachable' (bool), 'model' (str), 'provider' (str).
        """
        pass
