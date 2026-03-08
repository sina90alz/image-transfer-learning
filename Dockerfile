FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Install generic inference/runtime dependencies
COPY requirements.inference.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install CPU-only PyTorch explicitly
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /app/data

# Copy API code
COPY api /app/api

# Copy required model artifacts
RUN mkdir -p /app/artifacts
COPY artifacts/model.pt /app/artifacts/model.pt
COPY artifacts/classes.json /app/artifacts/classes.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host=0.0.0.0", "--port=8000"]