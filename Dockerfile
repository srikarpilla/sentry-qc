# ── Sentry QC Dockerfile — Google Cloud Run ──────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install system deps for Pillow / TF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY outputs/ ./outputs/
COPY index.html .

# Cloud Run injects PORT env var; default to 8080 if not set
ENV PORT=8080
EXPOSE 8080

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
