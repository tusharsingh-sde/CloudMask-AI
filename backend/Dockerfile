# ==============================================================================
# CloudMask AI - Production Backend Container (CPU-Optimized, Lean & Non-Root)
# ==============================================================================
FROM python:3.12-slim

# Prevent bytecode generation, buffer output, and disable pip caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install minimal OS runtime dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# 1. Install CPU-only PyTorch to avoid 2.5GB+ CUDA bloat
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install lean production requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application codebase
COPY . /app

# Optional build-time model baking (set via --build-arg DOWNLOAD_MODEL=true)
ARG DOWNLOAD_MODEL=false
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
        python download_model.py --skip-existing; \
    fi

# Ensure correct file permissions for non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Dependency-free container healthcheck targeting /api/ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/api/ready', timeout=5)" || exit 1

# Launch uvicorn dynamically honoring cloud platform PORT environment variables
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
