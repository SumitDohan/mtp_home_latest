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

# Copy only requirements first for caching
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src

# Create MLflow run directory (persist MLflow logs)
RUN mkdir -p /app/mlruns

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_PORT=5050 \
    FASTAPI_PORT=8000 \
    USE_DVC=false \
    TZ=Asia/Kolkata

# Expose ports for MLflow & FastAPI
EXPOSE ${FASTAPI_PORT} ${MLFLOW_PORT}

# Configure Git safe directory to avoid warnings
RUN git config --global --add safe.directory /app

# Clean __pycache__ before running pipeline
# Run pipeline components sequentially
# Keep MLflow UI & FastAPI running
CMD bash -c "\
echo '🧹 Cleaning cache directories...' && \
find /app/src -name '__pycache__' -exec rm -rf {} + && \
echo '📥 Running Ingestion...' && python src/components/ingestion.py && \
echo '🔄 Running Preprocessing...' && python src/components/preprocessing.py && \
echo '🧠 Running Model...' && python src/components/model.py && \
echo '📊 Running Feature Extraction...' && python src/components/feature.py && \
echo '🚀 Starting MLflow UI and FastAPI servers...' && \
mlflow ui --host 0.0.0.0 --port ${MLFLOW_PORT} & \
uvicorn src.api.app:app --host 0.0.0.0 --port ${FASTAPI_PORT}"
