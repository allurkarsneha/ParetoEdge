"""Measure every metrics.json field for one ONNX model, identically for any model.

Usage:
    python measure.py results/student_fp32.onnx --name baseline_fp32
    python measure.py results/student_int8.onnx --name int8 --skip-energy

Writes results/metrics/<name>.json. Run the baseline and every candidate with the
same flags on the same machine so the ratios are meaningful.

Method (PDF Section 5, applied to the ONNX model as Section 5.4 requires):
  - accuracy / worst class: ONNX Runtime over the val indices in val_split.json
  - params:   elements of all weight/bias initializers in the ONNX graph
              (QDQ scales/zero-points and int64 shape constants excluded)
  - MACs:     Conv + Gemm/MatMul multiply-accumulates from the shape-inferred graph
  - size:     on-disk ONNX file size
  - latency:  mean of timed single-image runs after warm-up
  - peak RAM: max RSS, sampled every 10 ms, in a fresh subprocess that loads only
              this model (so torch/other models do not inflate it)
  - energy:   CodeCarbon over a fixed-duration inference loop, divided by run count
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil

LATENCY_WARMUP = 20
LATENCY_RUNS = 200
RAM_RUNS = 50
ENERGY_SECONDS = 20
SEED = 42

# CodeCarbon CPU modes that read real power sensors; everything else is an estimate
MEASURED_CPU_MODES = {"intel_rapl", "intel_power_gadget", "apple_powermetrics", "windows_emi"}


def make_session(model_path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def dummy_input():
    return np.random.default_rng(SEED).standard_normal((1, 3, 224, 224)).astype(np.float32)


def ram_worker(model_path, threads):
    """Runs in its own process: peak RSS while loading the model and running inference."""
    process = psutil.Process(os.getpid())
    idle_rss_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = idle_rss_mb
    sampling = True

    def sample_rss(interval_s=0.01):
        nonlocal peak_rss_mb
        while sampling:
            peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 * 1024))
            time.sleep(interval_s)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    session = make_session(model_path, threads)
    input_name = session.get_inputs()[0].name
    x = dummy_input()
    for _ in range(RAM_RUNS):
        session.run(None, {input_name: x})
    sampling = False
    sampler.join()

    result = {"peak_rss_mb": peak_rss_mb, "idle_rss_mb": idle_rss_mb}
    if sys.platform != "win32":
        import resource

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        result["ru_maxrss_mb"] = max_rss / (1024 * 1024) if sys.platform == "darwin" else max_rss / 1024
    print(json.dumps(result))


def measure_peak_ram(model_path, threads):
    output = subprocess.run(
        [sys.executable, __file__, str(model_path), "--ram-worker", "--threads", str(threads)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(output.strip().splitlines()[-1])


def count_params(model):
    import onnx

    graph = model.graph
    qdq_aux = set()
    for node in graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            qdq_aux.update(node.input[1:])
    total = 0
    for tensor in graph.initializer:
        if tensor.name in qdq_aux or tensor.data_type == onnx.TensorProto.INT64:
            continue
        total += int(np.prod(tensor.dims)) if tensor.dims else 1
    return total


def count_macs(model):
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    shapes = {}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
    initializers = {t.name: list(t.dims) for t in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}

    def weight_shape(name):
        # In QDQ graphs the weight reaches Conv through a DequantizeLinear node
        if name in initializers:
            return initializers[name]
        producer = producers.get(name)
        if producer is not None and producer.op_type == "DequantizeLinear":
            return initializers[producer.input[0]]
        raise ValueError(f"Cannot resolve weight shape for {name}")

    total = 0
    for node in graph.node:
        if node.op_type == "Conv":
            out_shape = shapes[node.output[0]]
            w = weight_shape(node.input[1])  # [out_ch, in_ch / groups, kH, kW]
            total += int(np.prod(out_shape)) * w[1] * w[2] * w[3]
        elif node.op_type in ("Gemm", "MatMul"):
            total += int(np.prod(weight_shape(node.input[1])))  # batch size is 1
    return total


def measure_latency(session):
    input_name = session.get_inputs()[0].name
    x = dummy_input()
    for _ in range(LATENCY_WARMUP):
        session.run(None, {input_name: x})
    latencies = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def measure_accuracy(session, split_path):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)
    val_indices = json.loads(Path(split_path).read_text())["val"]
    loader = DataLoader(Subset(dataset, val_indices), batch_size=64, shuffle=False)

    input_name = session.get_inputs()[0].name
    num_classes = len(dataset.classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            # The exported graph has a fixed batch of 1, matching the required (1, 3, 224, 224)
            for image, label in zip(images.numpy(), labels.numpy()):
                logits = session.run(None, {input_name: image[None]})[0]
                confusion[label, int(logits.argmax())] += 1

    recall = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)
    return {
        "accuracy": float(confusion.diagonal().sum() / confusion.sum()),
        "worst_class_accuracy": float(recall.min()),
        "worst_class": dataset.classes[int(recall.argmin())],
        "per_class_accuracy": {name: float(r) for name, r in zip(dataset.classes, recall)},
        "confusion_matrix": confusion.tolist(),
        "evaluated_images": int(confusion.sum()),
    }


def measure_energy(session):
    from codecarbon import EmissionsTracker

    input_name = session.get_inputs()[0].name
    x = dummy_input()
    for _ in range(LATENCY_WARMUP):
        session.run(None, {input_name: x})

    # Process tracking attributes power to this process only, not other apps on the laptop
    tracker = EmissionsTracker(
        measure_power_secs=1, log_level="error", save_to_file=False, tracking_mode="process"
    )
    tracker.start()
    runs = 0
    deadline = time.perf_counter() + ENERGY_SECONDS
    while time.perf_counter() < deadline:
        session.run(None, {input_name: x})
        runs += 1
    tracker.stop()

    cpu_modes = [getattr(h, "_mode", None) for h in tracker._hardware if type(h).__name__ == "CPU"]
    measured = any(mode in MEASURED_CPU_MODES for mode in cpu_modes)
    return {
        "joules_per_inference": tracker.final_emissions_data.energy_consumed * 3.6e6 / runs,
        "energy_runs": runs,
        "codecarbon_mode": "measured" if measured else "tdp_estimate",
        "codecarbon_cpu_mode": cpu_modes[0] if cpu_modes else None,
    }


def machine_description():
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    else:
        cpu = platform.processor() or platform.machine()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return f"{cpu}, {ram_gb:.0f} GB, {platform.system()} {platform.release()}, CPU only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--name", help="output name (default: model file stem)")
    parser.add_argument("--split", default="val_split.json")
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--ram-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ram_worker:
        ram_worker(args.model, args.threads)
        return

    import onnx

    name = args.name or args.model.stem
    print(f"Measuring {args.model} as '{name}' (threads={args.threads})")

    onnx_model = onnx.load(str(args.model))
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 3, 224, 224], f"model must accept (1, 3, 224, 224), got {input_shape}"

    metrics = {
        "model": str(args.model),
        "size_mb": args.model.stat().st_size / (1024 * 1024),
        "params_million": count_params(onnx_model) / 1e6,
        "macs_giga": count_macs(onnx_model) / 1e9,
    }
    print(f"  size {metrics['size_mb']:.3f} MB | params {metrics['params_million']:.4f} M | MACs {metrics['macs_giga']:.4f} G")

    ram = measure_peak_ram(args.model, args.threads)
    metrics["peak_ram_mb"] = ram["peak_rss_mb"]
    metrics["ram_details"] = ram
    print(f"  peak RAM {ram['peak_rss_mb']:.1f} MB (process idle before load: {ram['idle_rss_mb']:.1f} MB)")

    session = make_session(args.model, args.threads)
    latency = measure_latency(session)
    metrics["latency_ms"] = latency["mean_ms"]
    metrics["latency_details"] = latency
    print(f"  latency mean {latency['mean_ms']:.3f} ms | median {latency['median_ms']:.3f} | p95 {latency['p95_ms']:.3f}")

    if not args.skip_accuracy:
        accuracy = measure_accuracy(session, args.split)
        metrics.update(accuracy)
        print(f"  accuracy {accuracy['accuracy'] * 100:.2f}% | worst class {accuracy['worst_class']} "
              f"{accuracy['worst_class_accuracy'] * 100:.2f}% ({accuracy['evaluated_images']} images)")

    if not args.skip_energy:
        energy = measure_energy(session)
        metrics.update(energy)
        print(f"  energy {energy['joules_per_inference']:.5f} J/inference over {energy['energy_runs']} runs "
              f"({energy['codecarbon_mode']}, cpu mode {energy['codecarbon_cpu_mode']})")

    metrics["settings"] = {
        "threads": args.threads, "split": args.split, "latency_runs": LATENCY_RUNS,
        "energy_seconds": ENERGY_SECONDS, "onnxruntime": ort.__version__,
    }
    metrics["measurement_machine"] = machine_description()

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
