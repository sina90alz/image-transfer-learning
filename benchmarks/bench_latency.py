import time
import argparse
import random
from pathlib import Path

import numpy as np
import requests


def percentile(values, p):
    return float(np.percentile(values, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/predict", help="endpoint url")
    ap.add_argument("--samples_dir", default="benchmarks/samples", help="folder with sample images")
    ap.add_argument("--n", type=int, default=100, help="number of timed requests")
    ap.add_argument("--warmup", type=int, default=10, help="warmup requests")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples dir not found: {samples_dir}")

    # Load images (limit to common image types)
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        images.extend(samples_dir.glob(ext))

    if len(images) < 3:
        raise ValueError(f"Need at least 3 images in {samples_dir}, found {len(images)}")

    print(f"Using {len(images)} sample images from: {samples_dir}")

    def call_once(img_path: Path):
        with open(img_path, "rb") as f:
            return requests.post(args.url, files={"file": f}, timeout=30)

    # Warmup (random images)
    for _ in range(args.warmup):
        img = random.choice(images)
        r = call_once(img)
        r.raise_for_status()

    # Timed requests (random images)
    times_ms = []
    picked = {p.name: 0 for p in images}

    for _ in range(args.n):
        img = random.choice(images)
        picked[img.name] += 1

        start = time.perf_counter()
        r = call_once(img)
        end = time.perf_counter()

        r.raise_for_status()
        times_ms.append((end - start) * 1000)

    p50 = percentile(times_ms, 50)
    p95 = percentile(times_ms, 95)
    avg = float(np.mean(times_ms))

    print("\nDistribution of chosen images:")
    for name, cnt in sorted(picked.items()):
        print(f"  {name}: {cnt}")

    print("\nLatency results (end-to-end HTTP):")
    print(f"Requests: {args.n}")
    print(f"Avg: {avg:.2f} ms")
    print(f"P50: {p50:.2f} ms")
    print(f"P95: {p95:.2f} ms")


if __name__ == "__main__":
    main()
