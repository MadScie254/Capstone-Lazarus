# CAPSTONE-LAZARUS: Multi-stage Docker Build
# Production-ready containerization with CPU/GPU support

# ============================================================================
# Base Stage: Common dependencies
# ============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    pkg-config \
    curl \
    wget \
    git \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Dependencies Stage: Install Python packages
# ============================================================================
FROM base as dependencies

# Copy requirements first (for better caching)
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Development Stage: For development and testing
# ============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets

# Copy source code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8501 8888 8000

# Default command for development
CMD ["streamlit", "run", "app/streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================================================
# Production Stage: Optimized for production
# ============================================================================
FROM dependencies as production

# Copy only necessary application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/

# Copy startup scripts
COPY scripts/docker-entrypoint.sh ./
COPY scripts/healthcheck.py ./

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Create directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/models /app/experiments && \
    chown -R appuser:appuser /app

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Expose port
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["streamlit"]

# ============================================================================
# GPU Stage: NVIDIA CUDA support
# ============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as gpu-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    pkg-config \
    curl \
    wget \
    git \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create app directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
COPY requirements-gpu.txt .

# Install Python dependencies with GPU support
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/
COPY scripts/docker-entrypoint.sh ./
COPY scripts/healthcheck.py ./

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Create directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/models /app/experiments && \
    chown -R appuser:appuser /app

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Expose port
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["streamlit"]

# ============================================================================
# Training Stage: Optimized for model training
# ============================================================================
FROM gpu-base as training

# Install additional training dependencies
USER root
RUN pip install --no-cache-dir \
    mlflow \
    wandb \
    optuna \
    ray[tune] \
    tensorboard \
    jupyterlab

# Copy training scripts
COPY scripts/train_model.py ./
COPY notebooks/ ./notebooks/

# Create directories for training artifacts
RUN mkdir -p /app/experiments /app/checkpoints /app/tensorboard && \
    chown -R appuser:appuser /app

USER appuser

# Expose additional ports for training services
EXPOSE 8501 6006 5000

# Default command for training
CMD ["python", "scripts/train_model.py"]

# ============================================================================
# Inference Stage: Optimized for model serving
# ============================================================================
FROM production as inference

# Install serving dependencies
USER root
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    gunicorn

# Copy FastAPI application
COPY app/fastapi/ ./app/fastapi/

USER appuser

# Expose FastAPI port
EXPOSE 8000

# Command for inference serving
CMD ["uvicorn", "app.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]