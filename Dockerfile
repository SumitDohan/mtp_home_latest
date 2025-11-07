# ============================================================
# ✅ Base image
# ============================================================
FROM python:3.10-slim

# ============================================================
# ✅ Set working directory and ensure root privileges
# ============================================================
WORKDIR /app
USER root

# ============================================================
# ✅ Install essential system dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    libsqlite3-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# ✅ Copy and install dependencies first (for better caching)
# ============================================================
COPY requirements.txt ./

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir dvc[s3] nltk mlflow uvicorn

# ============================================================
# ✅ Download required NLTK data
# ============================================================
RUN python -m nltk.downloader vader_lexicon

# ============================================================
# ✅ Copy full source code
# ============================================================
COPY src/ ./src

# ============================================================
# ✅ Create directories and fix permissions
# ============================================================
RUN mkdir -p /app/data/raw /app/data/processed /app/src /app/mlruns && \
    chmod -R 777 /app

# ============================================================
# ✅ Configure Git & DVC safe directories
# ============================================================
RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory /app/src

# ============================================================
# ✅ Environment variables
# ============================================================
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    MLFLOW_PORT=5050 \
    FASTAPI_PORT=8081 \
    USE_DVC=true \
    CI=false \
    TZ=Asia/Kolkata \
    DOCKER=true

# ============================================================
# ✅ Expose service ports
# ============================================================
EXPOSE ${FASTAPI_PORT} ${MLFLOW_PORT}

# ============================================================
# ✅ Entrypoint script
# ============================================================
CMD bash -c "\
echo '🧹 Cleaning cache directories...' && \
find /app/src -name '__pycache__' -exec rm -rf {} + && \

# --- DVC Handling ---
if [ \"$USE_DVC\" = \"true\" ]; then \
    echo '📥 Pulling DVC-tracked data...' && \
    dvc pull || (echo '⚠️ DVC pull failed — reinitializing...' && \
    dvc init --no-scm && echo '✅ DVC reinitialized (no-scm mode)'); \
else \
    echo 'ℹ️ USE_DVC=false — skipping DVC pull'; \
fi && \

# --- Auto-download missing CSVs ---
if [ ! -f /app/data/raw/news_NIFTY.csv ]; then \
    echo '📡 news_NIFTY.csv missing, downloading...' && \
    python src/components/ingestion.py --download-news-only || true; \
fi && \
if [ ! -f /app/data/raw/stock.csv ]; then \
    echo '📡 stock.csv missing, downloading...' && \
    python src/components/ingestion.py --download-stock-only || true; \
fi && \

# --- Run Pipeline ---
echo '📥 Running Ingestion...' && python src/components/ingestion.py || echo '⚠️ Ingestion failed, continuing...' && \
echo '🔄 Running Preprocessing...' && python src/components/preprocessing.py || echo '⚠️ Preprocessing failed, continuing...' && \
echo '🧠 Running Model...' && python src/components/model.py || echo '⚠️ Model run failed, continuing...' && \
echo '📊 Running Feature Extraction...' && python src/components/feature.py || echo '⚠️ Feature extraction failed, continuing...' && \

# --- Conditional Server Launch ---
if [ \"$CI\" != \"true\" ]; then \
    echo '🚀 Starting MLflow UI and FastAPI servers...' && \
    nohup mlflow ui --backend-store-uri file:///app/mlruns --host 0.0.0.0 --port ${MLFLOW_PORT} > mlflow.log 2>&1 & \
    uvicorn src.api.app:app --host 0.0.0.0 --port ${FASTAPI_PORT}; \
else \
    echo '✅ CI mode detected: skipping MLflow UI & FastAPI'; \
fi"
