#!/usr/bin/env bash
# Pruning + fine-tuning experiments behind the tradeoff curve (see report).
# Block 17 (s7) and the final conv are nearly free to prune (sensitivity analysis), so every
# candidate removes 75% of both; s2-s6 are pruned uniformly at the given ratio.
set -euo pipefail
PY=${PY:-edge-ai-env/bin/python}
FREE="--stage-ratios s7_7px=0.75 --last-ratio 0.75"

$PY -u prune.py run --name p00_free  $FREE --epochs 0
$PY -u prune.py run --name p00_free_ft_kd  $FREE --epochs 1 --kd
$PY -u prune.py run --name p25_ce    --ratio 0.25 $FREE --epochs 1
$PY -u prune.py run --name p25_kd    --ratio 0.25 $FREE --epochs 1 --kd
$PY -u prune.py run --name p40_ce    --ratio 0.40 $FREE --epochs 1
$PY -u prune.py run --name p40_kd    --ratio 0.40 $FREE --epochs 1 --kd
