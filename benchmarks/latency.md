# Latency Benchmark

## Setup
- API: FastAPI (Docker)
- Model: EfficientNet-B0 (Transfer Learning)
- Endpoint: POST /predict
- Image size: 224
- Top-k: 3
- Batch size: 1 (single-image inference)
- Device: CPU
- Host OS: Windows (Docker Desktop / WSL2)

## Method
- Warmup: 20 requests
- Timed requests: 300
- Load pattern: sequential requests, random image selection from 3 samples (NORMAL/PNEUMONIA/TB)
- Measurement: end-to-end HTTP latency (client → API → response)
- Metrics: Avg, P50, P95

## Results

### CPU
- Avg: 30.82 ms
- P50: 31.43 ms
- P95: 37.86 ms

## Notes
- P95 reflects tail latency and is important for production readiness.
- Future improvements: batching at model level, TorchScript serving, ONNX runtime, concurrency testing.
