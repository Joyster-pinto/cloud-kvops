# Cloud-KVOps: Production Dockerfile for AWS EC2 / App Runner / ECS
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages (using CPU torch by default for lightweight cloud instances)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose FastAPI server port (8000) and Streamlit port (8501)
EXPOSE 8000 8501

# Default command: launch the FastAPI Dual-Streaming Telemetry server
CMD ["python", "server.py"]
