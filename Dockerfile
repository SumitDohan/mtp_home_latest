# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Create necessary directories
RUN mkdir -p /app/mlruns /app/data/raw /app/data/processed

# Add non-root user and switch
RUN useradd -ms /bin/bash mluser
USER mluser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_PORT=5050 \
    FASTAPI_PORT=8000 \
    USE_DVC=true \
    CI=false \
    TZ=Asia/Kolkata

# Expose ports
EXPOSE ${FASTAPI_PORT} ${MLFLOW_PORT}

# Configure Git safe directory
RUN git config --global --add safe.directory /app

# Entrypoint: run pipeline
CMD bash -c "\
echo '🧹 Cleaning cache directories...' && \
find /app/src -name '__pycache__' -exec rm -rf {} + && \
# Pull DVC-tracked data if enabled \
if [ \"$USE_DVC\" = \"true\" ]; then \
    echo '📥 Pulling DVC-tracked data...' && dvc pull || echo '⚠️ DVC pull failed, continuing...'; \
fi && \
# Auto-download missing CSV files \
[ ! -f /app/data/raw/news_NIFTY.csv ] && echo '📡 news_NIFTY.csv missing, downloading...' && python src/components/ingestion.py --download-news-only || true; \
[ ! -f /app/data/raw/stock.csv ] && echo '📡 stock.csv missing, downloading...' && python src/components/ingestion.py --download-stock-only || true; \
# Run full ingestion \
echo '📥 Running Ingestion...' && python src/components/ingestion.py || echo '⚠️ Ingestion failed, continuing...'; \
# Run preprocessing \
echo '🔄 Running Preprocessing...' && python src/components/preprocessing.py || echo '⚠️ Preprocessing failed, continuing...'; \
# Run model \
echo '🧠 Running Model...' && python src/components/model.py || echo '⚠️ Model run failed, continuing...'; \
# Run feature extraction \
echo '📊 Running Feature Extraction...' && python src/components/feature.py || echo '⚠️ Feature extraction failed, continuing...'; \
# Start MLflow UI and FastAPI if not CI \
if [ \"$CI\" != \"true\" ]; then \
    echo '🚀 Starting MLflow UI and FastAPI servers...' && \
    mlflow ui --host 0.0.0.0 --port ${MLFLOW_PORT} & \
    uvicorn src.api.app:app --host 0.0.0.0 --port ${FASTAPI_PORT}; \
else \
    echo '✅ CI mode detected: pipeline completed, skipping MLflow UI & FastAPI'; \
fi"
