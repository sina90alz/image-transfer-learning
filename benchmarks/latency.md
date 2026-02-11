# Latency Benchmark

## Setup
- API: FastAPI (Docker)
- Model: EfficientNet-B0 (Transfer Learning)
- Endpoint: `POST /predict`
- Hardware:
  - CPU: <fill>
  - GPU: <fill if available>
- Image size: 224
- Top-k: 3

## Method
- Warmup: 10 requests
- Timed requests: 100
- Measurement: end-to-end HTTP latency (client → API → response)
- Metrics: Avg, P50, P95

## Results

### CPU
- Avg: <fill> ms
- P50: <fill> ms
- P95: <fill> ms

### GPU (optional)
- Avg: <fill> ms
- P50: <fill> ms
- P95: <fill> ms

## Notes
- P95 reflects tail latency and is important for production readiness.
- Future improvements: batching at model level, TorchScript serving, ONNX runtime, concurrency testing.
