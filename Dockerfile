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
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only src folder
COPY src/ ./src

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run scripts in components folder
CMD ["bash", "-c", "python src/components/ingestion.py && python src/components/preprocessing.py && python src/components/news_sentiment.py && python src/components/feature.py"]
