#!/usr/bin/env bash
# =============================================================================
# Croply AI Service - Ollama Initialization & Auto-pull Entrypoint
# Inicia Ollama en background, verifica el modelo gemma4 y lo descarga si falta.
# =============================================================================

set -e

MODEL_NAME="${OLLAMA_MODEL:-gemma4:e4b}"

echo "[Ollama-Entrypoint] Iniciando servidor Ollama en segundo plano..."
ollama serve &
PID=$!

echo "[Ollama-Entrypoint] Esperando que la API de Ollama esté disponible en 127.0.0.1:11434..."
until curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[Ollama-Entrypoint] Servidor Ollama listo y respondiendo."

# Verificar si el modelo configurado ya se encuentra descargado
if ollama list | grep -q "${MODEL_NAME}"; then
    echo "[Ollama-Entrypoint] El modelo '${MODEL_NAME}' ya se encuentra disponible en caché local."
else
    echo "[Ollama-Entrypoint] Modelo '${MODEL_NAME}' no encontrado localmente. Iniciando descarga automática (pull)..."
    ollama pull "${MODEL_NAME}"
    echo "[Ollama-Entrypoint] Modelo '${MODEL_NAME}' descargado y verificado exitosamente."
fi

echo "[Ollama-Entrypoint] Ollama operativo y listo para atender solicitudes de Croply."
wait $PID
