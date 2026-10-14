"""Sweep INT8 calibration settings on an FP32 ONNX model and measure each result.

Usage:
    python quant_sweep.py                                   # sweeps the baseline FP32 model
    python quant_sweep.py --input results/pruned_fp32.onnx --tag pruned

Energy is skipped during the sweep to save time; measure the chosen model fully with
measure.py afterwards. Results go to results/quant_sweep_<tag>.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from quantize import quantize

# (method, calibration images, keep first conv + classifier in FP32)
CONFIGS = [
    ("minmax", 100, False),
    ("minmax", 500, False),
    ("minmax", 1000, False),
    ("percentile", 500, False),
    ("minmax", 500, True),
    ("entropy", 200, False),  # ORT entropy calibration is very slow; 500 ran >10 min
]

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/student_fp32.onnx")
parser.add_argument("--tag", default="baseline")
args = parser.parse_args()

out_dir = Path("results/quant_sweep") / args.tag
rows = []
for method, calib_size, keep_fp32 in CONFIGS:
    name = f"{args.tag}_int8_{method}_{calib_size}" + ("_fp32ends" if keep_fp32 else "")
    model_path = out_dir / f"{name}.onnx"
    metrics_path = Path(f"results/metrics/{name}.json")
    print(f"\n=== {name}")
    if model_path.exists() and metrics_path.exists():
        print("  already done, reusing results")  # lets an interrupted sweep resume
    else:
        quantize(args.input, model_path, method, calib_size, keep_fp32)
        subprocess.run([sys.executable, "measure.py", str(model_path), "--name", name, "--skip-energy"], check=True)
    metrics = json.loads(metrics_path.read_text())
    rows.append({
        "name": name, "method": method, "calib_size": calib_size, "keep_first_last_fp32": keep_fp32,
        **{k: metrics[k] for k in ("accuracy", "worst_class_accuracy", "worst_class",
                                   "size_mb", "latency_ms", "peak_ram_mb")},
    })

Path(f"results/quant_sweep_{args.tag}.json").write_text(json.dumps(rows, indent=2))

print(f"\n{'config':45s} {'acc%':>7s} {'worst%':>7s} {'worst class':>22s} {'MB':>6s} {'ms':>6s} {'RAM':>6s}")
for r in rows:
    print(f"{r['name']:45s} {r['accuracy'] * 100:7.2f} {r['worst_class_accuracy'] * 100:7.2f} "
          f"{r['worst_class']:>22s} {r['size_mb']:6.2f} {r['latency_ms']:6.2f} {r['peak_ram_mb']:6.1f}")
print(f"Saved: results/quant_sweep_{args.tag}.json")
