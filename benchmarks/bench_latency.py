import time
import argparse
import numpy as np
import requests
from pathlib import Path

def percentile(values, p):
    return float(np.percentile(values, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/predict", help="endpoint url")
    ap.add_argument("--image", required=True, help="path to image file")
    ap.add_argument("--n", type=int, default=100, help="number of requests")
    ap.add_argument("--warmup", type=int, default=10, help="warmup requests")
    args = ap.parse_args()

    img_path = Path(args.image)
    assert img_path.exists(), f"Image not found: {img_path}"

    # warmup
    for _ in range(args.warmup):
        with open(img_path, "rb") as f:
            requests.post(args.url, files={"file": f})

    times_ms = []
    for _ in range(args.n):
        start = time.perf_counter()
        with open(img_path, "rb") as f:
            r = requests.post(args.url, files={"file": f})
        end = time.perf_counter()

        r.raise_for_status()
        times_ms.append((end - start) * 1000)

    p50 = percentile(times_ms, 50)
    p95 = percentile(times_ms, 95)
    avg = float(np.mean(times_ms))

    print(f"Requests: {args.n}")
    print(f"Avg: {avg:.2f} ms")
    print(f"P50: {p50:.2f} ms")
    print(f"P95: {p95:.2f} ms")

if __name__ == "__main__":
    main()
