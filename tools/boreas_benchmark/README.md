# Boreas-Objects-V1 benchmark

Prepare labelled Boreas scans, generate six model configs, evaluate local 3D
AP40 and measure batch-one latency. Local validation is carved from official
training windows; this is not the official hidden-test or a weather benchmark.

## Prepare and verify

From the repository root:

```bash
python tools/boreas_benchmark/test_benchmark.py
python tools/boreas_benchmark/selftest.py
python tools/boreas_benchmark/prepare.py prepare --beams 128
python tools/boreas_benchmark/make_configs.py
```

Prepare 64/32/16 in the same way for reduced-beam experiments. Frame IDs must
match the 128-beam split. Whole official training windows are split using every
fifth window for validation, with a two-second purge. The exporter keeps Misc,
while the models learn Car, Pedestrian and Cyclist. Shared augmentation: x/y
flips, yaw rotation +/-45 degrees, scale 0.95--1.05; no database sampling.

## Training seed, precision and checkpoints

Run from `tools`. Example for a NEW FP32 experiment:

```bash
python train.py --cfg_file cfgs/boreas_models/pv_rcnn.yaml \
  --batch_size 1 --epochs 80 --workers 2 \
  --fix_random_seed --seed 42 --no_amp \
  --extra_tag seed42_fp32_v2 --range_eval
```

The resulting tag is `seed42_fp32_v2_beam128`. Replace the config stem for other
models. `--seed` defaults to 42 and controls model initialization and training
workers when `--fix_random_seed` is present. Without that flag, training is not
seeded. Old runs using that flag used seed 666 even if named `seed42`.
Never relabel old checkpoints as seed 42. To reproduce that old seed use
`--fix_random_seed --seed 666`.

AMP is optional: `--use_amp` enables it; `--no_amp` explicitly disables it even
if enabled in YAML. PV-RCNN's current PointNet2 kernels require FP32 in the
failing grouping path; AMP support has not been added by these benchmark fixes.
Mixed training precisions can be disclosed, but use the same inference precision
for accuracy and efficiency comparisons.

Existing checkpoints in a tag cause a fresh run to stop. Use a NEW tag for a new
seed/precision/batch/LR experiment, or explicitly pass `--resume` or `--ckpt` to
continue. For resuming old training retain its original seed and settings.
Each invocation records arguments, resolved config, actual batch size, seed and
precision in `run_config_<timestamp>.json`; it does not overwrite old checkpoints.

All configs default to 80 epochs and batch 1/GPU. CLI batch size overrides YAML;
choose the same feasible actual batch and number of GPUs for the main comparison.
LR (0.003 for Pillars/SECOND/CenterPoint, 0.001 VoxelNeXt, 0.01 PV-RCNN), gradient
clipping, losses, architecture, NMS IoU thresholds and pillar/voxel representation
remain model-specific and must be disclosed. Use the same tuning budget.
Use epoch 80 for all models, or predefine the same checkpoint-selection rule.
Do not mix last-epoch and best-validation results without stating the rule.

## Inference budget and AP protocol

Generated and current Boreas configs use score cutoff 0.001, pre-NMS cap 4096,
and final NMS cap 500. VoxelNeXt no longer keeps the 83-detection template cap.
PV-RCNN's internal ROI proposal stage remains architecture-specific. Prediction
scores below 0.001 are still unavailable; this is an explicit practical AP cutoff,
not a guarantee of an untruncated precision--recall curve.

With `--range_eval`, `accuracy.json` and `range_ap.json` report local AP40 v1:
3D rotated IoU >=0.7 for Car, >=0.5 for Pedestrian/Cyclist; score-descending,
one-to-one greedy matching; interpolation at recall 1/40 through 1. Unmatched
predictions overlapping Misc at the class IoU threshold are ignored for all target
classes. This is a declared LOCAL convention, not an official Boreas score.
GT and prediction centers are independently filtered into radial bins [0,20),
[20,40), [40,60), [60,infinity). Boundary crossing can therefore affect bin AP.
Precision/recall/F1 and TP/FP/FN are reported separately at score 0.1.

KITTI-derived metrics are saved separately in `kitti_accuracy.json`. KITTI
conversion creates artificial 2D boxes and difficulty attributes, so bbox AP and
Easy/Moderate/Hard are not real Boreas difficulty measurements. KITTI maps Misc
to Van and has different matching/ignore semantics. Do not mix its AP with local
AP40 in the same primary comparison. Without `--range_eval`, the primary local
accuracy report explicitly has no AP result. Existing report consumers must use
`mAP40` for the new primary report instead of the old `mAP_R40` key.

## Re-evaluate existing checkpoints without training

Pass the EXACT existing output folder tag; this need not be the true seed name.
Run from `tools`:

```bash
bash boreas_benchmark/run_metrics.sh pointpillar seed42 80 42
```

Use `seed42_beam128` instead if that is the folder containing the checkpoint.
The script tests the SAME checkpoint on 128/64/32/16 beams, measures FP32 on
128 beams, and summarizes models with that tag. This is test-time sparsity
robustness, not retraining on each beam count. To train and validate at another
beam count, pass `--boreas_beams 64` to train and use the corresponding checkpoint.

Single 128-beam evaluation:

```bash
python test.py --cfg_file cfgs/boreas_models/pointpillar.yaml \
  --ckpt ../output/boreas_models/pointpillar/seed42/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 --workers 2 --seed 42 --extra_tag seed42 --range_eval
```

Re-export predictions with the updated configs before recomputing AP. Saved
`result.pkl` from score cutoff 0.1 cannot recover omitted low-score detections.

## Timing and summary

```bash
python boreas_benchmark/profile.py --cfg cfgs/boreas_models/pointpillar.yaml \
  --ckpt ../output/boreas_models/pointpillar/seed42/ckpt/checkpoint_epoch_80.pth \
  --boreas_beams 128 --seed 42 --frames 200 --warmup 20 \
  --out ../output/boreas_models/pointpillar/seed42/profile.json
python boreas_benchmark/summarize.py --tag seed42 --out ../output/boreas_summary/seed42
```

No `--amp` means FP32 inference. Optional AMP profiling is a separate experiment.
Use the same GPU, software, seed, frames, warmup and precision for every model.
The profiler records TF32 settings and refuses CUDA_LAUNCH_BLOCKING=1.
It reports synchronized full-forward time (including head decoding and ALL NMS),
end-to-end time, mean/p50/p95, FPS, peak memory and parameter count. End-to-end
starts at loading prepared .npy, includes CPU preprocessing and transfer, and
excludes raw .bin decoding, sensor capture, rendering and result serialization.
Warmup populates caches, so disk I/O is not a cold-cache benchmark.

No network/NMS split is reported: CenterPoint/VoxelNeXt execute NMS inside their
heads, so timing just the detector post_processing method would mislabel it.
`test.py --infer_time` is synchronized and skips warmup batches; use profile.py
for the paper. The summarizer rejects mismatched metric versions, GPU, inference
precision or TF32 settings; regenerate legacy reports instead of mixing them.
Three seeds and repeated latency measurements strengthen close comparisons.
