# ParetoEdge

Compressing a distilled MobileNetV2 (EuroSAT land-use classification) for edge deployment.

## Hackathon
Built for **The Minimalist Edge AI Challenge: Pruning, Quantizing, and Distilling Heavy Models for Lean, Green Edge Hardware**. This was the **Software Optimization Track**, a one-day team competition held on October 13, 2026 at the **11th ACM/IEEE Symposium on Edge Computing (SEC 2026)**, Santa Clara University.

**Challenge task:**
- The organizers provided a MobileNetV2 already distilled from a ResNet-50 teacher and trained on [EuroSAT](https://github.com/phelber/EuroSAT): 27,000 Sentinel-2 satellite images in 10 land-use classes.
- The goal was to compress it further for edge hardware while keeping the (1, 3, 224, 224) input, measuring every step on CPU, and submitting a quantized ONNX model.

**Scoring (100 points):**
- Model accuracy and per-class consistency: 25
- Compression (parameters and file size): 20
- Resource efficiency (latency and peak RAM, as ratios against the team's own baseline): 20
- Energy per inference (as a ratio against the team's own baseline): 20
- Reproducibility: 10
- Engineering judgment (tradeoff analysis): 5

The organizers' checkpoints and the competition brief aren't included in this repository.

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
