FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for large file processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    libssl-dev \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create only essential directories (data will be mounted as volumes)
RUN mkdir -p /app/logs && \
    chmod 755 /app && \
    chmod 755 /app/logs

# Set environment variables for large file processing
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MALLOC_ARENA_MAX=2

# Copy only application code (exclude data, models, uploads via .dockerignore)
COPY . .

# Set memory limits for large file processing
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Set environment variables for external volumes
ENV TRANSFORMERS_CACHE=/app/models/transformers_cache
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

EXPOSE 8000

CMD ["python", "run.py"]