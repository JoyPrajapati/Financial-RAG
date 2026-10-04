FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first — prevents pulling CUDA version
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ ./api/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Download and bake embedding model into image at build time
# This means no internet needed at runtime
RUN python scripts/download_model.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]