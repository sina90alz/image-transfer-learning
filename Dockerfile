FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# install inference runtime deps
COPY requirements.inference.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy API package + model
COPY api /app/api
COPY artifacts/model.pt /app/artifacts/model.pt
COPY artifacts/classes.json /app/artifacts/classes.json

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host=0.0.0.0", "--port=8000"]