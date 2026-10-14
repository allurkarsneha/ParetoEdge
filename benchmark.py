import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = {
    "FP32": "results/student_fp32.onnx",
    "INT8": "results/student_int8.onnx"
}

# Same input for both models
rng = np.random.default_rng(42)
dummy_input = rng.standard_normal(
    (1, 3, 224, 224)
).astype(np.float32)

results = {}

for model_name, model_path in MODELS.items():
    print(f"\nBenchmarking {model_name}...")

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Fix thread settings for a consistent comparison
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(20):
        session.run(None, {input_name: dummy_input})

    latencies = []

    for _ in range(100):
        start = time.perf_counter()

        session.run(None, {input_name: dummy_input})

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    size_mb = path.stat().st_size / (1024 * 1024)

    results[model_name] = {
        "model_size_mb": round(size_mb, 3),
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "median_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "inference_runs": 100,
        "device": "Mac local CPU",
        "execution_provider": "CPUExecutionProvider",
        "intra_op_threads": 1
    }

    print(f"Model size: {size_mb:.3f} MB")
    print(f"Mean latency: {np.mean(latencies):.3f} ms")
    print(f"Median latency: {np.median(latencies):.3f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.3f} ms")

# Calculate improvements
fp32 = results["FP32"]
int8 = results["INT8"]

size_reduction = (
    (fp32["model_size_mb"] - int8["model_size_mb"])
    / fp32["model_size_mb"]
) * 100

speedup = (
    fp32["mean_latency_ms"]
    / int8["mean_latency_ms"]
)

results["comparison"] = {
    "size_reduction_percent": round(size_reduction, 2),
    "latency_speedup": round(speedup, 3)
}

Path("results").mkdir(exist_ok=True)

with open("results/comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== PARETOEDGE COMPARISON =====")
print(f"Size reduction: {size_reduction:.2f}%")
print(f"Latency speedup: {speedup:.3f}x")
print("\nSaved: results/comparison.json")