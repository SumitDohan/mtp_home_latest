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

# Copy src folder
COPY src/ ./src

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:/app/mlruns
ENV MLFLOW_PORT=5050        
ENV FASTAPI_PORT=8000       

# Expose ports
EXPOSE ${FASTAPI_PORT} ${MLFLOW_PORT}

# Entrypoint: run pipeline + MLflow + FastAPI
CMD bash -c "\
if [ ! -d /app/mlruns ]; then mkdir /app/mlruns; fi && \
python src/components/ingestion.py && \
python src/components/preprocessing.py && \
python src/components/model.py && \
python src/components/feature.py && \
mlflow ui --host 0.0.0.0 --port ${MLFLOW_PORT} & \
uvicorn src.api.app:app --host 0.0.0.0 --port ${FASTAPI_PORT}"
