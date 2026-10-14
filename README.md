# ParetoEdge

Compressing a distilled MobileNetV2 (EuroSAT land-use classification) for edge deployment.
Built for the Minimalist Edge AI Challenge at SEC 2026, software optimization track.

**Result:** static QDQ INT8 quantization with Percentile calibration. The model is 3.4× smaller (8.51 → 2.49 MB), runs 1.7× faster (8.96 → 5.16 ms) and uses 45% less energy per inference, at 98.81% accuracy (baseline 99.33%).
The full write-up is in [submission/REPORT.md](submission/REPORT.md).

## Layout
| Path | Purpose |
|---|---|
| `make_split.py` | Seeded, stratified 80/20 train/val split → `val_split.json` |
| `main.py` | Load the provided student checkpoint and export FP32 ONNX |
| `quantize.py` | Static QDQ INT8 quantization with stratified train-set calibration |
| `quant_sweep.py` | Compares calibration methods and sizes |
| `measure.py` | Accuracy, worst class, params, MACs, size, latency, peak RAM, energy for any ONNX model |
| `prune.py` | Structured channel pruning: sensitivity analysis, KD fine-tuning (investigated, not shipped) |
| `mixed_precision.py` | Layer-wise INT8/FP32 search (implemented, not run) |
| `profile_aihub.py` | Qualcomm AI Hub compile and profile |
| `submission/` | Submitted model, metrics.json, report, tradeoff curve |

## Reproduce
Not in this repo: the dataset (`python download_data.py` → `./data`) and the organizer-provided checkpoints, which go in `models/`.
```
python -m venv edge-ai-env && source edge-ai-env/bin/activate
pip install -r requirements.txt
python main.py
python quantize.py --method percentile --calib-size 500 --output results/student_int8.onnx
python measure.py results/student_fp32.onnx --name baseline_fp32
python measure.py results/student_int8.onnx --name final_int8_percentile
```
