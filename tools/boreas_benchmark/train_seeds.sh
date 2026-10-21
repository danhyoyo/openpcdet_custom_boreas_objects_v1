#!/usr/bin/env bash
# Run from tools. Example: bash boreas_benchmark/train_seeds.sh second 43 44
set -euo pipefail
MODEL=${1:?Pass a model name}; shift
if [ "$#" -eq 0 ]; then set -- 42 43 44; fi
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_PREFIX=${RUN_PREFIX:-seed}
for SEED in "$@"; do
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer"; exit 1; }
  TAG="${RUN_PREFIX}${SEED}_beam128"
  CKPT_DIR="../output/boreas_models/$MODEL/$TAG/ckpt"
  if [ -d "$CKPT_DIR" ] && find "$CKPT_DIR" -maxdepth 1 -name '*.pth' -print -quit | grep -q .; then
    echo "$MODEL/$TAG already contains checkpoints; keep it and request only missing seeds."
    exit 1
  fi
  "$PYTHON_BIN" train.py --cfg_file "cfgs/boreas_models/$MODEL.yaml" \
    --batch_size 4 --epochs 80 --workers 2 --fix_random_seed --seed "$SEED" \
    --use_amp --boreas_beams 128 --extra_tag "${RUN_PREFIX}${SEED}" --range_eval
done
