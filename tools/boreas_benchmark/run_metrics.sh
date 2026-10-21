#!/usr/bin/env bash
# Evaluate saved checkpoints with modulo or angular-profile sampling.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."
MODEL=${1:-pointpillar}
TAG=${2:-seed42_beam128}
EPOCH=${3:-80}
SEED=${4:-42}
# MODE=modulo preserves the original four-density benchmark.
# MODE=elevation prepares angular datasets and evaluates the SAME checkpoint.
# MODE=elevation-plan only reads training scans and prints the mapping.
# MODE=multiphase evaluates every valid modulo offset for 64/32/16 rings and
# writes phase-wise AP plus mean/std/min/max summaries for the SAME checkpoint.
# MODE=all runs the complete paper suite: modulo, elevation, and multiphase.
MODE=${5:-modulo}
PYTHON=${BOREAS_PYTHON:-python}
DATA_ROOT=${BOREAS_DATA_ROOT:-/home/ducanh/datasets/boreas_openpcdet}
CACHE_ROOT=${BOREAS_DENSITY_CACHE:-../output/boreas_density_cache}
if [[ "$MODE" != modulo && "$MODE" != elevation && "$MODE" != elevation-plan && "$MODE" != multiphase && "$MODE" != all ]]; then
  echo "Mode must be modulo, multiphase, elevation, elevation-plan, or all (argument 5)." >&2
  exit 1
fi
if [[ "$MODE" == all ]]; then
  echo "Running complete paper evaluation suite for $MODEL/$TAG"
  "$SCRIPT_DIR/run_metrics.sh" "$MODEL" "$TAG" "$EPOCH" "$SEED" modulo
  "$SCRIPT_DIR/run_metrics.sh" "$MODEL" "$TAG" "$EPOCH" "$SEED" elevation
  "$SCRIPT_DIR/run_metrics.sh" "$MODEL" "$TAG" "$EPOCH" "$SEED" multiphase
  echo "Complete paper evaluation suite finished for $MODEL/$TAG"
  exit 0
fi
if [[ "$MODE" == elevation || "$MODE" == elevation-plan ]]; then
  read -r -a PROFILES <<< "${BOREAS_ELEVATION_PROFILES:-vlp16 vlp32c}"
  if [[ ${#PROFILES[@]} == 0 ]]; then
    echo "Select at least one elevation profile." >&2
    exit 1
  fi
  ELEV_ROOT=${BOREAS_ELEVATION_DATA_ROOT:-$DATA_ROOT/elevation}
  ELEV_ARGS=(--raw "${BOREAS_RAW_ROOT:-/home/ducanh/datasets/boreas/boreas-objects-v1}"
    --reference "$DATA_ROOT/128" --destination "$ELEV_ROOT"
    --catalog "${BOREAS_ELEVATION_CATALOG:-$SCRIPT_DIR/lidar_profiles.json}"
    --calibration-frames "${BOREAS_CALIBRATION_FRAMES:-20}"
    --max-angle-error "${BOREAS_MAX_ANGLE_ERROR:-2}"
    --profiles "${PROFILES[@]}")
  if [[ "${BOREAS_FOV_CONTROLS:-1}" != 0 ]]; then
    ELEV_ARGS+=(--with-fov-controls)
  fi
  if [[ "$MODE" == elevation-plan ]]; then
    "$PYTHON" boreas_benchmark/elevation.py plan "${ELEV_ARGS[@]}"
    exit 0
  fi
  BASE="../output/boreas_models/$MODEL/$TAG"
  CKPT="$BASE/ckpt/checkpoint_epoch_$EPOCH.pth"
  CFG="cfgs/boreas_models/$MODEL.yaml"
  test -f "$CKPT" || { echo "Checkpoint missing: $CKPT" >&2; exit 1; }
  test -f "$CFG" || { echo "Config missing: $CFG" >&2; exit 1; }
  "$PYTHON" boreas_benchmark/elevation.py prepare "${ELEV_ARGS[@]}"
  CONDITIONS=()
  for PROFILE in "${PROFILES[@]}"; do
    CONDITIONS+=("$PROFILE")
    if [[ "${BOREAS_FOV_CONTROLS:-1}" != 0 ]]; then
      CONDITIONS+=("${PROFILE}_dense_fov")
    fi
  done
  # Check all datasets before any GPU pass; unchanged GT, IDs and point hashes.
  COUNTS=()
  for CONDITION in "${CONDITIONS[@]}"; do
    BEAM=$("$PYTHON" boreas_benchmark/elevation.py check \
      --data "$ELEV_ROOT/$CONDITION" --reference "$DATA_ROOT/128")
    COUNTS+=("$BEAM")
  done
  for INDEX in "${!CONDITIONS[@]}"; do
    CONDITION=${CONDITIONS[$INDEX]}
    BEAM=${COUNTS[$INDEX]}
    DATA="$ELEV_ROOT/$CONDITION"
    EVAL_TAG="elevation_$CONDITION"
    RESULT="$BASE/eval/epoch_$EPOCH/val/$EVAL_TAG"
    REPORT="$BASE/elevation/epoch_$EPOCH/$CONDITION"
    echo "Evaluating $MODEL: $CONDITION ($BEAM distinct source rings)"
    "$PYTHON" test.py --cfg_file "$CFG" --ckpt "$CKPT" --batch_size 1 --workers 2 \
      --seed "$SEED" --extra_tag "$TAG" --eval_tag "$EVAL_TAG" --range_eval \
      --set DATA_CONFIG.DATA_PATH "$DATA"
    cp "$DATA/manifest.json" "$RESULT/sampling_manifest.json"
    mkdir -p "$REPORT"
    "$PYTHON" boreas_benchmark/extended.py --predictions "$RESULT/result.pkl" \
      --data "$DATA" --reference-data "$DATA_ROOT/128" --beam "$BEAM" \
      --cache-dir "$CACHE_ROOT" --out "$REPORT/extended.json"
  done
  "$PYTHON" boreas_benchmark/elevation.py summarize --base "$BASE" --epoch "$EPOCH"
  echo "Elevation metrics complete. Reports are separate from modulo results and the beam-only workbook."
  exit 0
fi
if [[ "$MODE" == multiphase ]]; then
  BASE="../output/boreas_models/$MODEL/$TAG"
  CKPT="$BASE/ckpt/checkpoint_epoch_$EPOCH.pth"
  CFG="cfgs/boreas_models/$MODEL.yaml"
  SOURCE_ROOT=${BOREAS_SOURCE_ROOT:-/home/ducanh/datasets/boreas}
  test -f "$CKPT" || { echo "Checkpoint missing: $CKPT" >&2; exit 1; }
  test -f "$CFG" || { echo "Config missing: $CFG" >&2; exit 1; }
  test -f "$DATA_ROOT/128/ImageSets/val.txt" || { echo "Dense validation split missing" >&2; exit 1; }

  # There are 2, 4, and 8 valid modulo phases for 64, 32, and 16 rings.
  # Phase zero reuses the legacy 64/32/16 paths; nonzero phases have separate
  # prepared datasets so that no existing condition is overwritten.
  for BEAM in 64 32 16; do
    MODULUS=$((128 / BEAM))
    for ((PHASE=0; PHASE<MODULUS; PHASE++)); do
      if [[ "$PHASE" -eq 0 ]]; then
        CONDITION="$BEAM"
      else
        CONDITION="${BEAM}_phase${PHASE}"
      fi
      DATA="$DATA_ROOT/$CONDITION"
      if [[ ! -f "$DATA/manifest.json" || ! -f "$DATA/custom_infos_val.pkl" ]]; then
        "$PYTHON" boreas_benchmark/prepare.py prepare \
          --source "$SOURCE_ROOT" --destination "$DATA_ROOT" \
          --beams "$BEAM" --phase "$PHASE"
      fi
      cmp -s "$DATA_ROOT/128/ImageSets/val.txt" "$DATA/ImageSets/val.txt" || {
        echo "Validation frame IDs differ for $CONDITION" >&2
        exit 1
      }
    done
  done

  for BEAM in 64 32 16; do
    MODULUS=$((128 / BEAM))
    for ((PHASE=0; PHASE<MODULUS; PHASE++)); do
      if [[ "$PHASE" -eq 0 ]]; then
        CONDITION="$BEAM"
        EVAL_TAG="beam$BEAM"
      else
        CONDITION="${BEAM}_phase${PHASE}"
        EVAL_TAG="beam${BEAM}_phase${PHASE}"
      fi
      DATA="$DATA_ROOT/$CONDITION"
      RESULT="$BASE/eval/epoch_$EPOCH/val/$EVAL_TAG"
      echo "Evaluating $MODEL: $BEAM rings, modulo phase $PHASE/$((MODULUS - 1))"
      "$PYTHON" test.py --cfg_file "$CFG" --ckpt "$CKPT" \
        --batch_size 1 --workers 2 --seed "$SEED" --extra_tag "$TAG" \
        --eval_tag "$EVAL_TAG" --range_eval \
        --set DATA_CONFIG.DATA_PATH "$DATA"
      cp "$DATA/manifest.json" "$RESULT/sampling_manifest.json"
      "$PYTHON" boreas_benchmark/extended.py \
        --predictions "$RESULT/result.pkl" --data "$DATA" \
        --reference-data "$DATA_ROOT/128" --beam "$BEAM" \
        --cache-dir "$CACHE_ROOT" --out "$BASE/extended_${CONDITION}.json"
    done
  done

  "$PYTHON" boreas_benchmark/phase_summary.py \
    --eval-root "$BASE/eval/epoch_$EPOCH/val" \
    --data-root "$DATA_ROOT" --out "$BASE/phase_summary"
  echo "Multiphase metrics complete: $BASE/phase_summary"
  exit 0
fi
WORKBOOK=${BOREAS_EXCEL_PATH:-../output/boreas_summary/boreas_benchmark_results.xlsx}
"$PYTHON" boreas_benchmark/export_excel.py --check-dependency
BASE="../output/boreas_models/$MODEL/$TAG"
CKPT="$BASE/ckpt/checkpoint_epoch_$EPOCH.pth"
CFG="cfgs/boreas_models/$MODEL.yaml"
test -f "$CKPT" || { echo "Checkpoint missing: $CKPT. Pass the exact existing output tag as argument 2."; exit 1; }
test -f "$CFG" || { echo "Config missing: $CFG"; exit 1; }
for BEAM in 128 64 32 16; do
  DATA="$DATA_ROOT/$BEAM"
  test -f "$DATA/custom_infos_val.pkl" || { echo "Prepare $BEAM beams first"; exit 1; }
  cmp -s "$DATA_ROOT/128/ImageSets/val.txt" "$DATA/ImageSets/val.txt" || { echo "Validation frame IDs differ at $BEAM beams"; exit 1; }
done
for BEAM in 128 64 32 16; do
  "$PYTHON" test.py --cfg_file "$CFG" --ckpt "$CKPT" --batch_size 1 --workers 2 --seed "$SEED" --extra_tag "$TAG" --eval_tag "beam$BEAM" --range_eval --set DATA_CONFIG.DATA_PATH "$DATA_ROOT/$BEAM"
  RESULT="$BASE/eval/epoch_$EPOCH/val/beam$BEAM"
  "$PYTHON" boreas_benchmark/extended.py --predictions "$RESULT/result.pkl" --data "$DATA_ROOT/$BEAM" --reference-data "$DATA_ROOT/128" --beam "$BEAM" --cache-dir "$CACHE_ROOT" --out "$BASE/extended_$BEAM.json"
  if [ "$BEAM" = 128 ]; then
    cp "$RESULT/range_ap.json" "$BASE/range_ap.json"
    cp "$RESULT/accuracy.json" "$BASE/accuracy.json"
    cp "$RESULT/kitti_accuracy.json" "$BASE/kitti_accuracy.json"
  else
    cp "$RESULT/range_ap.json" "$BASE/range_ap_$BEAM.json"
  fi
done
# Export the same frame from each test pass, before profiling/workbook export.
if [ "${BOREAS_VISUALIZE:-1}" != 0 ]; then
  VIS_ARGS=(--frame-index "${BOREAS_VIS_FRAME_INDEX:-0}")
  if [ -n "${BOREAS_VIS_FRAME_ID:-}" ]; then
    VIS_ARGS=(--frame-id "$BOREAS_VIS_FRAME_ID")
  fi
  "$PYTHON" boreas_benchmark/visualize_rings.py --eval-root "$BASE/eval/epoch_$EPOCH/val" \
    --score-thresh "${BOREAS_VIS_SCORE:-0.3}" "${VIS_ARGS[@]}" || {
    echo "Warning: image export failed. Metrics will continue; rerun visualize_rings.py after fixing the reported issue." >&2
  }
fi
# No --amp: full forward (including every head's NMS) and end-to-end are FP32.
# Profile the actual sparse inputs; never infer their latency from ring count.
PROFILE_REPEATS=${BOREAS_PROFILE_REPEATS:-1}
for BEAM in 128 64 32 16; do
  "$PYTHON" boreas_benchmark/profile.py --cfg "$CFG" --ckpt "$CKPT" \
    --boreas_beams "$BEAM" --seed "$SEED" --frames 200 --warmup 20 \
    --repeats "$PROFILE_REPEATS" --out "$BASE/profile_$BEAM.json"
done
# Backward-compatible dense-profile name for existing study/export tooling.
cp "$BASE/profile_128.json" "$BASE/profile.json"
if [ "${BOREAS_EXPORT_SENSITIVITY:-0}" != 0 ]; then
  "$PYTHON" boreas_benchmark/sensitivity.py \
    --predictions "$BASE/eval/epoch_$EPOCH/val/beam128/result.pkl" \
    --infos "$DATA_ROOT/128/custom_infos_val.pkl" \
    --out "$BASE/sensitivity_128.json"
fi
# Commit this completed model/seed to the shared workbook before cross-model summaries.
"$PYTHON" boreas_benchmark/export_excel.py --model "$MODEL" --tag "$TAG" --epoch "$EPOCH" --workbook "$WORKBOOK"
"$PYTHON" boreas_benchmark/summarize.py --root ../output/boreas_models --tag "$TAG" --out "../output/boreas_summary/$TAG"
