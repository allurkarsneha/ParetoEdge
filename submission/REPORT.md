# ParetoEdge: Technical Report

**Team:** ParetoEdge · **Submitted model:** `model_int8.onnx` (static QDQ INT8, input (1, 3, 224, 224)) · **Machine:** Apple M4, 16 GB, ONNX Runtime CPU, 1 thread

## Result (same machine, same script for baseline and final)

| Metric | FP32 baseline | Final INT8 | Ratio |
|---|---|---|---|
| Val accuracy (5,400 images) | 99.33% | 98.81% | 0.995 |
| Worst class | 96.40% (PermanentCrop) | 96.00% (PermanentCrop) | |
| Params | 2.22 M | 2.22 M | 1.00 |
| ONNX size | 8.51 MB | 2.49 MB | **0.29** |
| Latency | 8.96 ms | 5.16 ms | **0.58** |
| Peak RAM | 89.3 MB | 77.7 MB | 0.87 |
| Energy (CodeCarbon, TDP estimate) | 0.108 J | 0.059 J | **0.55** |

## Method and measurement
- **Split:** the organizers provided no `val_split.json`, so `make_split.py` makes a seeded (42), class-stratified 80/20 split (21,600 train / 5,400 val). Calibration and pruning statistics use train images only. All accuracy figures are on val.
- **Measurement (`measure.py`):** all metrics come from the exported ONNX graph under ONNX Runtime.
  - Params and MACs are counted from the graph.
  - Peak RAM is sampled every 10 ms in a fresh process.
  - Energy comes from a 20 s inference loop divided by the run count, because 100 runs is shorter than CodeCarbon's 1 s sampling interval. CodeCarbon fell back to `tdp_estimate` (cpu_load mode, no sudo powermetrics).

## What we tried, kept, and dropped

**1. Static INT8 calibration sweep (kept: Percentile, 500 images).** See `results/quant_sweep_baseline.json` and the left panel of `tradeoff_curve.png`.

| Calibration | Accuracy | Worst class |
|---|---|---|
| MinMax 100 / 500 / 1000 images | 98.30 / 98.22 / 98.52% | 94.75 / 94.00 / 96.50% |
| **Percentile 500** | **98.81%** | 96.00% |
| MinMax 500 + first conv & classifier in FP32 | 98.52% | **97.00%** |
| Entropy 200 | 98.44% | 96.00% |

- **Our first INT8 attempt was calibrated on dataset indices 0–199, which are all one class (AnnualCrop).** We replaced it with stratified train sampling.
- **Entropy calibration at 500 images ran for more than 10 minutes on CPU without beating Percentile, so we dropped it.**
- Percentile clips activation outliers, giving the best mean accuracy at no size cost. Keeping the first and last layers in FP32 gave the best worst class (+1 pt) for +0.04 MB. We chose Percentile because accuracy carries more points than the 1-point worst-class gain.

**2. Structured channel pruning (investigated, not shipped).** See `prune.py`, `results/prune_sensitivity_stats.json` and the right panel of `tradeoff_curve.png`. We pruned the hidden (expansion) channels of the inverted-residual blocks; this never breaks a skip connection.
- **Negative result:** a weight-only importance score was no better than random on some blocks. At 25% pruning of stage s2 it dropped accuracy to 55%.
- **Fix:**
  - Rank channels by activation std × outgoing weight norm, measured on 256 train images.
  - Fold each removed channel's mean activation into the next BatchNorm or classifier bias (bias compensation).
  - The same 25% prune of s2 then kept 96.6% accuracy with no fine-tuning (s5: 61.7% → 97.6%).
- **The cliff is not in the same place for every layer:**
  - Stages s2–s6 survive 25% (96–98.7%) but collapse at 50% (66–75%).
  - Block 17 and the final 1280-channel conv lose under 0.3 pt even at 75%.
- **Pruning effects are not additive:** pruning both "free" layers together (2.24M → 1.56M params) gave 93.3% (SeaLake 58.5%) before fine-tuning.
- **Fine-tuning was not completed in time.** Fine-tuning could recover that loss, but PyTorch on this CPU trains MobileNetV2 at about 5 img/s (about 72 min per epoch). On the Apple GPU (MPS) it is about 67 img/s, and that run ran out of hackathon time. Following the PDF's advice that a completed ordinary result beats an unfinished one, we shipped the quantized unpruned model.

**3. Knowledge distillation (prepared, not evaluated).** Teacher logits are cached for the train split. The ResNet-50 teacher scores 98.8% on train, below the student's 99.3% val accuracy, so distillation may transfer the teacher's errors. Hard-label vs KD fine-tuning is implemented (`prune.py run --kd`) but not evaluated.

**4. Layer-wise mixed precision (implemented, not run).** `mixed_precision.py` ranks each Conv/Gemm by how much keeping only that layer in FP32 reduces logit error, then sweeps how many layers stay FP32.

## Qualcomm AI Hub
Not run: the API token was not configured in time. All figures are local measurements (PDF Section 7).

## Next steps
1. Fine-tune `p25` with s7 and the final conv at 75% (about 1.2M params) on MPS, then quantize.
2. Re-compute pruning statistics after each stage, to address the non-additive interaction.
3. Mixed-precision sweep, then profile on AI Hub.
