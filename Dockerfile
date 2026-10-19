# =============================================================================
# Croply AI Service - Production Dockerfile
# Python 3.13-slim + LightGBM (libgomp1) + FastAPI
# =============================================================================

FROM python:3.13-slim AS production

# Variables de entorno para optimización de Python y logs limpios
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Instalación de librerías del sistema requeridas:
# - libgomp1: Soporte OpenMP para cálculo multihilo de LightGBM
# - curl: Utilidad para Docker Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Creación de usuario no privilegiado por seguridad en producción
RUN groupadd -g 10001 croplygroup && \
    useradd -m -u 10001 -g croplygroup -s /bin/bash croplyuser

WORKDIR /app

# Copia e instalación de dependencias productivas
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia del código de la aplicación y modelos
COPY --chown=croplyuser:croplygroup app/ ./app/

# Cambio al usuario no root
USER croplyuser

# Exposición del puerto del microservicio
EXPOSE 8000

# Verificación periódica de estado del servicio
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/ai/health || exit 1

# Comando de arranque del servidor ASGI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
