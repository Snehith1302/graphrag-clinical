FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and required runtime dependencies
COPY backend /app
COPY data/vector_index /app/data/vector_index
COPY ingestion /app/ingestion

# Expose backend port
EXPOSE 8000

# Set environment path to root python packages
ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
