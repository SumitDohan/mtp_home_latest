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

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Create MLflow run directory (persist MLflow logs)
RUN mkdir -p /app/mlruns

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_PORT=5050 \
    FASTAPI_PORT=8000 \
    USE_DVC=true \
    TZ=Asia/Kolkata

# Expose ports
EXPOSE ${FASTAPI_PORT} ${MLFLOW_PORT}

# Configure Git safe directory to avoid warnings
RUN git config --global --add safe.directory /app

# Pipeline entrypoint: pull DVC data, run pipeline, start servers
CMD bash -c "\
echo '🧹 Cleaning cache directories...' && \
find /app/src -name '__pycache__' -exec rm -rf {} + && \
if [ \"$USE_DVC\" = \"true\" ]; then \
    echo '📥 Pulling DVC-tracked data...' && dvc pull; \
fi && \
echo '📥 Running Ingestion...' && python src/components/ingestion.py && \
echo '🔄 Running Preprocessing...' && python src/components/preprocessing.py && \
echo '🧠 Running Model...' && python src/components/model.py && \
echo '📊 Running Feature Extraction...' && python src/components/feature.py && \
echo '🚀 Starting MLflow UI and FastAPI servers...' && \
mlflow ui --host 0.0.0.0 --port ${MLFLOW_PORT} & \
uvicorn src.api.app:app --host 0.0.0.0 --port ${FASTAPI_PORT}"
