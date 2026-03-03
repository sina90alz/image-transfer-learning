FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app/inference

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

COPY inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY inference /app/inference

# Copy model checkpoint (adjust path if needed)
COPY inference/model.pt /app/model.pt

EXPOSE 8000

CMD ["uvicorn", "inference.app.main:app", "--host=0.0.0.0", "--port=8000"]