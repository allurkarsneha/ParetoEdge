# ParetoEdge: reproduce the submission

Submitted model: `model_int8.onnx` (input `input`, shape (1, 3, 224, 224)). Metrics are in `metrics.json`. The method and results are in `REPORT.md`.

## Setup
Python 3.11. Run these from a project root that contains `models/student_mobilenetv2_distilled.pt`, the EuroSAT data in `./data` (`datasets.EuroSAT(root="./data")`) and the files from `code/`.
```
pip install torch torchvision onnx onnxscript onnxruntime codecarbon psutil matplotlib
```
Versions used: torch 2.14.1, torchvision 0.29.1, onnx 1.23.2, onnxruntime 1.31.0, codecarbon 3.3.1.

## Pipeline
```
python make_split.py                 # writes val_split.json; a copy is already in code/
python main.py                       # exports results/student_fp32.onnx (opset 13, dynamo=False)
python quantize.py --method percentile --calib-size 500 --output results/student_int8.onnx   # final model
python measure.py results/student_fp32.onnx --name baseline_fp32
python measure.py results/student_int8.onnx --name final_int8_percentile
```
- `quant_sweep.py` reproduces the calibration comparison.
- `python prune.py sensitivity` reproduces the pruning sensitivity analysis.
- `run_prune_experiments.sh` and `mixed_precision.py` are the unfinished experiments described in the report.
