"""Layer-wise mixed-precision (INT8 / FP32) quantization search.

1. Calibrate once (ranges cached, so every variant below uses identical ranges).
2. Per-layer sensitivity: for each Conv/Gemm layer, build the all-INT8 model with only that
   layer left in FP32 and measure how much the output logits move back toward FP32
   (MSE vs FP32 logits on 256 train images). Bigger recovery = more quantization-sensitive.
3. Sweep N: keep the N most sensitive layers in FP32, quantize the rest, measure each variant
   with measure.py. More FP32 layers usually means better accuracy but bigger, slower models,
   which is the tradeoff we report.

Usage:
    python mixed_precision.py --input results/student_fp32.onnx --tag baseline
    python mixed_precision.py --input results/pruned/p25_kd_fp32.onnx --tag p25_kd --counts 0 2 4 8
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from quantize import (
    CALIBRATION_METHODS,
    EuroSATCalibrationReader,
    stratified_calibration_indices,
    transform,
)

SEED = 42
SENSITIVITY_IMAGES = 256


def quantize_variant(preprocessed, output, reader_factory, method, cache_path, exclude):
    quantize_static(
        model_input=str(preprocessed),
        model_output=str(output),
        calibration_data_reader=None if cache_path.exists() else reader_factory(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CALIBRATION_METHODS[method],
        nodes_to_exclude=exclude,
        calibration_cache_path=str(cache_path),
    )


def logits(model_path, images):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return np.concatenate([session.run(None, {input_name: image[None]})[0] for image in images])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/student_fp32.onnx")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--method", choices=CALIBRATION_METHODS, default="percentile")
    parser.add_argument("--calib-size", type=int, default=500)
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16],
                        help="numbers of most-sensitive layers to keep in FP32")
    parser.add_argument("--split", default="val_split.json")
    args = parser.parse_args()

    out_dir = Path("results/mixed_precision") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessed = out_dir / "preprocessed.onnx"
    cache_path = out_dir / f"calibration_{args.method}_{args.calib_size}.json"
    quant_pre_process(args.input, str(preprocessed))

    dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)
    train_indices = json.loads(Path(args.split).read_text())["train"]
    calibration_indices = stratified_calibration_indices(dataset, train_indices, args.calib_size)
    input_name = onnx.load(str(preprocessed)).graph.input[0].name

    def reader_factory():
        loader = DataLoader(Subset(dataset, calibration_indices), batch_size=1, shuffle=False)
        return EuroSATCalibrationReader(loader, input_name)

    # Sensitivity probe images: train images not used for calibration
    rng = np.random.default_rng(SEED + 1)
    probe_pool = sorted(set(train_indices) - set(calibration_indices))
    probe_indices = rng.choice(probe_pool, SENSITIVITY_IMAGES, replace=False).tolist()
    probe_images = np.stack([dataset[i][0].numpy() for i in probe_indices])
    reference = logits(args.input, probe_images)

    layers = [n.name for n in onnx.load(str(preprocessed)).graph.node if n.op_type in ("Conv", "Gemm", "MatMul")]
    print(f"{len(layers)} quantizable layers; calibrating ({args.method}, {args.calib_size} images)...")

    with tempfile.TemporaryDirectory() as tmp:
        all_int8 = Path(tmp) / "all_int8.onnx"
        quantize_variant(preprocessed, all_int8, reader_factory, args.method, cache_path, [])
        base_error = float(np.mean((logits(all_int8, probe_images) - reference) ** 2))
        print(f"all-INT8 logit MSE vs FP32: {base_error:.5f}")

        sensitivity = []
        for i, layer in enumerate(layers):
            variant = Path(tmp) / "variant.onnx"
            quantize_variant(preprocessed, variant, reader_factory, args.method, cache_path, [layer])
            error = float(np.mean((logits(variant, probe_images) - reference) ** 2))
            sensitivity.append({"layer": layer, "logit_mse_when_fp32": error, "recovery": base_error - error})
            print(f"  [{i + 1}/{len(layers)}] {layer:45s} recovery {base_error - error:+.5f}")

    sensitivity.sort(key=lambda row: row["recovery"], reverse=True)
    ranked = [row["layer"] for row in sensitivity]

    rows = []
    for count in args.counts:
        name = f"{args.tag}_mp_{args.method}_fp32x{count}"
        model_path = out_dir / f"{name}.onnx"
        quantize_variant(preprocessed, model_path, reader_factory, args.method, cache_path, ranked[:count])
        subprocess.run([sys.executable, "measure.py", str(model_path), "--name", name,
                        "--skip-energy", "--split", args.split], check=True)
        metrics = json.loads(Path(f"results/metrics/{name}.json").read_text())
        rows.append({"name": name, "fp32_layers": ranked[:count], "count": count,
                     **{k: metrics[k] for k in ("accuracy", "worst_class_accuracy", "worst_class",
                                                "size_mb", "latency_ms", "peak_ram_mb", "params_million")}})

    preprocessed.unlink()
    result = {"input": args.input, "method": args.method, "calib_size": args.calib_size,
              "all_int8_logit_mse": base_error, "sensitivity": sensitivity, "variants": rows}
    out = Path(f"results/mixed_precision_{args.tag}.json")
    out.write_text(json.dumps(result, indent=2))

    print(f"\nMost sensitive layers: {ranked[:5]}")
    print(f"{'FP32 layers':>11s} {'acc%':>7s} {'worst%':>7s} {'MB':>6s} {'ms':>6s}")
    for r in rows:
        print(f"{r['count']:11d} {r['accuracy'] * 100:7.2f} {r['worst_class_accuracy'] * 100:7.2f} "
              f"{r['size_mb']:6.2f} {r['latency_ms']:6.2f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
