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
if enabled in YAML. PV-RCNN's PointNet2 ball-query and grouping CUDA kernels require FP32. Their
autocast boundaries now convert floating inputs to FP32 and run backward in the
same precision; surrounding PointNet MLPs, backbone and heads can still use AMP.
The ROI classification loss now uses stable sigmoid+BCE fusion (BCEWithLogits),
which is supported by autocast; ignored labels stay masked. No CUDA extension
rebuild is needed for these Python fixes.
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
each of those four prepared inputs, and summarizes models with that tag. This is test-time sparsity
robustness, not retraining on each beam count. To train and validate at another
beam count, pass `--boreas_beams 64` to train and use the corresponding checkpoint.

Single 128-beam evaluation:

```bash
python test.py --cfg_file cfgs/boreas_models/pointpillar.yaml \
  --ckpt ../output/boreas_models/pointpillar/seed666_beam128/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 --workers 2 --seed 42 --extra_tag seed42 --range_eval
```

Re-export predictions with the updated configs before recomputing AP. Saved
`result.pkl` from score cutoff 0.1 cannot recover omitted low-score detections.

## Timing and summary

```bash
python boreas_benchmark/profile.py --cfg cfgs/boreas_models/pointpillar.yaml \
  --ckpt ../output/boreas_models/pointpillar/seed666_beam128/ckpt/checkpoint_epoch_80.pth \
  --boreas_beams 128 --seed 42 --frames 200 --warmup 20 --repeats 3 \
  --out ../output/boreas_models/pointpillar/seed666_beam128/profile_128.json
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
`run_metrics.sh` writes `profile_128.json`, `profile_64.json`, `profile_32.json`,
and `profile_16.json`, plus the legacy dense alias `profile.json`. Set
`BOREAS_PROFILE_REPEATS=3` for three passes over the same deterministic frame
sample. The summarizer writes `sparsity_efficiency.csv` and beam-specific Pareto
figures with measured point/cell counts; it never scales dense latency by ring count.

### AP export sensitivity (no training)

Saved predictions can test stricter score cutoffs and additional global
per-frame truncation of the exported detections:

```bash
python boreas_benchmark/sensitivity.py \
  --predictions ../output/boreas_models/second/seed666_beam128/eval/epoch_80/val/beam128/result.pkl \
  --infos /home/ducanh/datasets/boreas_openpcdet/128/custom_infos_val.pkl \
  --out ../output/boreas_models/second/seed666_beam128/sensitivity_128.json
```

Set `BOREAS_EXPORT_SENSITIVITY=1` when running `run_metrics.sh` to generate this
audit automatically. The tool verifies the prediction hash and source export
manifest. This global post-export truncation is not necessarily identical to
each head's configured pre/post-NMS cap semantics. A configured cap change,
lower cutoff, or higher cap requires fresh inference with an explicitly changed
config. The audit is CPU-only and does not retrain a model.

### Multiple modulo phases (no training)

Phase zero preserves the existing `64`, `32`, and `16` directories. Additional
phases use separate paths such as `32_phase1`:

```bash
python boreas_benchmark/prepare.py prepare --beams 32 --phase 1
python test.py --cfg_file cfgs/boreas_models/second.yaml \
  --ckpt ../output/boreas_models/second/seed666_beam128/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 --workers 2 --seed 42 --extra_tag seed666_beam128 \
  --eval_tag beam32_phase1 --range_eval \
  --set DATA_CONFIG.DATA_PATH /home/ducanh/datasets/boreas_openpcdet/32_phase1
```

For 64/32/16 rings, valid phases are respectively `0..1`, `0..3`, and `0..7`.
Every manifest records the modulo, phase, and exact retained ring IDs. Run all
valid phases with the same frozen checkpoint and report mean, sample standard
deviation, minimum, and maximum AP; do not present phase zero as phase-robustness.
After every phase has been evaluated, aggregate one frozen checkpoint with:

```bash
python boreas_benchmark/phase_summary.py \
  --eval-root ../output/boreas_models/second/seed666_beam128/eval/epoch_80/val \
  --out ../output/boreas_models/second/seed666_beam128/phase_summary
```

The aggregator requires the complete phase matrix by default and rejects a
checkpoint/config/export mismatch across phases. `--allow-partial` is for
exploration only and marks the JSON report incomplete.

The complete preparation, inference, extended-metric, and aggregation sequence
is available as one `run_metrics.sh` mode:

```bash
bash boreas_benchmark/run_metrics.sh second seed666_beam128 80 42 multiphase
```

To regenerate the complete metric suite used by the paper in one command, run:

```bash
bash boreas_benchmark/run_metrics.sh second seed666_beam128 80 666 all
```

The `all` mode runs modulo evaluation and profiling, elevation-matched profiles
with their FOV controls, and the complete multiphase evaluation in that order.

This evaluates all 2/4/8 phases for 64/32/16 rings with one frozen checkpoint.
It writes per-phase reports beside the standard evaluation outputs and writes
`phase_summary.json`, `phase_scores.csv`, and `phase_aggregates.csv` under the
run's `phase_summary` directory. It does not retrain or profile latency for
each phase. Use `BOREAS_SOURCE_ROOT` and `BOREAS_DATA_ROOT` to override the raw
and prepared dataset roots.

No network/NMS split is reported: CenterPoint/VoxelNeXt execute NMS inside their
heads, so timing just the detector post_processing method would mislabel it.
`test.py --infer_time` is synchronized and skips warmup batches; use profile.py
for the paper. The summarizer rejects mismatched metric versions, GPU, inference
precision or TF32 settings; regenerate legacy reports instead of mixing them.
Three seeds and repeated latency measurements strengthen close comparisons.

## PV-RCNN AMP training

Use a NEW tag when changing the precision or seed of an existing experiment:

```bash
cd /home/ducanh/projects/openPCDet/tools
# Prefer the existing AI environment packages over incompatible ~/.local NumPy.
export PYTHONPATH="/home/ducanh/miniforge3/envs/AI/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python train.py --cfg_file cfgs/boreas_models/pv_rcnn.yaml \
  --batch_size 2 --epochs 80 --workers 2 \
  --fix_random_seed --seed 42 --use_amp \
  --extra_tag seed42_amp_v3 --range_eval
```

Use the same flags and tag for the other five model configs. Do not combine
`--use_amp` with `--no_amp`. Optional CUDA regression and real-data smoke check:

```bash
python boreas_benchmark/test_pvrcnn_amp.py --model-smoke --batch-size 2 --steps 2
```

This checks PointNet grouping outputs/gradients, half-precision coordinates,
AMP-enabled MLPs, and a short real-data training/inference pass; it saves no
checkpoints. The smoke uses a modest loss scale for checking numerical paths;
normal training retains its configured GradScaler behavior. Model support does
not establish accuracy or throughput over a full 80-epoch run.
The metric runner still uses FP32 inference for a common evaluation protocol;
training AMP and inference FP32 are independent choices and must be reported.

Validation on the RTX 5060 Ti: four CUDA operator/loss regression tests and a
real Boreas batch-2 AMP smoke with two optimizer steps passed (finite loss and
gradients; AMP inference finite). Peak allocated memory was about 3.21 GiB for
these smoke frames; this is not a bound for all training batches.
The successful test used NumPy 1.26.4 already installed in the AI environment.
NumPy 2.2.6 in ~/.local was overriding it and caused a separate native failure
inside cumm's from_numpy before model execution. The PYTHONPATH setting above
selects the existing compatible environment without installing/removing packages;
use the same environment for all six models. This does not prove that all cumm
builds are incompatible with NumPy 2.

## Checkpoint frequency

Training now defaults to saving one checkpoint at the end of each epoch
(`--ckpt_save_interval 1`) and disables timed mid-epoch `latest_model.pth` saves
(`--ckpt_save_time_interval 0`). The progress output every 50 iterations is only
logging and does not save checkpoints. Existing checkpoints are not deleted by
this change. The existing retention policy still keeps the newest 30 epoch
checkpoints by default; use `--max_ckpt_save_num 80` to retain all 80 epochs.
To explicitly enable timed backups later, pass a positive seconds interval.
Changes apply to newly started/restarted processes, not an already running loop.

## Extended density / confidence / rank / three-seed evaluation (2026-10-08)

Training architecture, augmentation and existing checkpoints are unchanged.
`run_metrics.sh` now runs full 128/64/32/16 inference, primary AP, extended
CPU evaluation and dense-input profiling. It creates `extended_128.json`,
`extended_64.json`, `extended_32.json`, `extended_16.json` in each run folder,
plus per-class reliability SVG files. First use computes GT point counts;
progress prints every 25 frames and later models reuse shared caches in
`output/boreas_density_cache`. No dataset files need rewriting.

```bash
cd /home/ducanh/projects/openPCDet/tools
bash boreas_benchmark/run_metrics.sh pointpillar seed42 80 42
bash boreas_benchmark/run_metrics.sh second seed42_beam128 80 42
# Run the same command for centerpoint / voxelnext / pv_rcnn / pointpillar_lite
# only once the requested epoch checkpoint exists.
```

The final argument is the evaluation seed, NOT an assertion of the historical
training seed. Fresh inference saves `inference_manifest.json` with actual
cutoff, final caps, checkpoint/model/prediction hashes, precision and TF32.
Legacy prediction files lacking this evidence are not silently certified.

CPU-only re-analysis of predictions exported under the new manifest protocol:

```bash
python boreas_benchmark/extended.py \
  --predictions ../output/boreas_models/second/seed666_beam128/eval/epoch_80/val/beam16/result.pkl \
  --data /home/ducanh/datasets/boreas_openpcdet/16 --beam 16 \
  --out ../output/boreas_models/second/seed666_beam128/extended_16.json
```

`--allow-unverified` permits exploratory legacy analysis, marks it unverified,
and is rejected by the audited study. It cannot recover scores removed by old
cutoffs. Default calibration uses 15 equal-width bins, post-NMS native scores,
class-specific one-to-one 3D TP/FP matching and the local Misc ignore rule.
ECE/Brier are fractions, not percentages. Report recall and AP alongside them:
missed objects have no detection confidence. No calibrator is fitted.

Density recall uses point counts within GT boxes before model-specific voxel
caps. Groups: 0, 1–5, 6–20, 21–50, 51–100, 101+. `density_fixed128` preserves
cohort identity across beams; `density_actual` regroups by each condition's
count. Full-domain matches are stratified by class, GT-center range and density.
This is recall, not AP obtained by discarding other ground-truth cohorts.

For current runs with different folder tags, edit the example CSV to contain
only runs with completed metric reports, then use:

```bash
python boreas_benchmark/summarize.py \
  --runs boreas_benchmark/current_runs.example.csv \
  --out ../output/boreas_summary/current
```

Outputs include `ranking_stability.json` (average ranks and Kendall tau-b),
`calibration.csv`, `density_recall.csv`, accuracy/range/beam tables and Pareto
SVGs. This one-run summary is DESCRIPTIVE; it does not certify a common seed.
Missing basic reports are skipped. Undefined tau for fewer than two models or
all ties is null. The reported model intersection is explicit.

### Independent training seeds

```bash
# Example: add missing independent runs without overwriting seed42 checkpoints.
bash boreas_benchmark/train_seeds.sh second 43 44
# With no seed arguments it requests 42,43,44; aborts if a tag has checkpoints.
# Same prospective protocol: batch4, workers2, 80 epochs, AMP, 128 beams.
```

After each new seed finishes, run `run_metrics.sh` using its tag and evaluation
seed, e.g. `bash boreas_benchmark/run_metrics.sh second seed43_beam128 80 43`.
Seed names are metadata, not evidence: older runs whose actual seed is unknown
can still be evaluated, but cannot be silently included as a known training seed.

```bash
python boreas_benchmark/study.py \
  --runs boreas_benchmark/runs.example.csv --seeds 42 43 44 \
  --out ../output/boreas_summary/multiseed
```

Edit `runs.example.csv` to the exact tags and selected model set. It lists all
six configurations by default; remove Lite rows only if it is excluded from
the experiment. A complete three-seed study requires 18 training runs for six
configurations, or 15 for the five established models. Existing audited runs can
be reused. CPU aggregation does not start training/inference.

`study.py` verifies actual training seed, batch, precision, epochs, worker count,
split/augmentation/features, per-model training settings across seeds, full
four-beam report protocol, fixed weights, and dense FP32 profile hardware/settings.
Identical weights cannot count as different seeds. It exports seed values,
mean/sample std with valid counts, density/calibration mean/std, rankings across
beam/range/seed, top-rank frequency, mean Pareto plots and seed-wise frontier
membership frequency. The mean frontier is descriptive, not a confidence bound.

New training processes save split hashes. For older metadata with a verified
seed but no historical split hashes, `--allow-legacy-splits` explicitly assumes
unchanged partitions and retains provenance warnings. It does not invent a seed
or override an actual split/hash mismatch. The optional `metadata` CSV column
selects an exact run_config JSON when a folder contains several runs.
`--allow-partial` marks incomplete studies exploratory; it never bypasses missing
seed metadata or mismatched conditions. Three test passes of one checkpoint do
not constitute three training seeds. Interval bootstrap remains future work.

CPU regressions (no GPU job):

```bash
cd /home/ducanh/projects/openPCDet
python3 -m unittest discover -s tools/boreas_benchmark -p 'test_[be]*.py'
```

If an existing seed-named folder has an unknown historical seed, keep it and
start a verifiable independent run under a fresh prefix, for example:

```bash
RUN_PREFIX=study_seed bash boreas_benchmark/train_seeds.sh second 42 43 44
```

This writes `study_seed42_beam128`, etc.; edit the CSV to these exact tags.
It does not replace the legacy folders. Request only missing verified seeds
when reusing a known existing run.

### Legacy run folder names (2026-10-09)

The completed PointPillars, SECOND and CenterPoint legacy runs are now stored
under `seed666_beam128`. Their original logs and checkpoints are preserved.
The seed is inferred from the historical fixed-seed implementation documented
above, rather than recorded in an original run_config file; each renamed run
contains `legacy_run_rename.json` describing that provenance.
PV-RCNN remains under `seed42_beam128`. Evaluation may consistently use seed 42
regardless of a checkpoint's training seed.


## Shared Excel workbook after each completed run

`run_metrics.sh` now updates `output/boreas_summary/boreas_benchmark_results.xlsx`
after all four beam evaluations and dense FP32 profiling succeed. Runtime export
uses `openpyxl` in the same WSL Python environment as evaluation; the desktop
artifact-tool/Node runtime is not installed in that Linux environment. If needed:

```bash
python -m pip install openpyxl
```

Sheets: `Runs` (accuracy/efficiency and actual training settings), `Accuracy`
(per class/range/beam and P/R/F1), `Calibration` (ECE/Brier), `Density`
(points-per-object recall), and `Reliability` (confidence-bin support and means).
Values are numeric; AP is percent, AP drops are percentage points, other rates
are fractions. Missing values stay blank. JSON reports remain authoritative.
Training seed and evaluation seed are distinct columns. Legacy seeds documented
by `legacy_run_rename.json` are marked `inferred_legacy`; inclusion in Excel does
not bypass the stricter multi-seed `study.py` provenance checks.

Updates replace the existing model/training-seed/epoch rows across all managed
sheets; other runs remain. Atomic replacement and a Linux file lock avoid partial
files/lost concurrent writes. Close the workbook in Excel before updating it.
The exporter refuses incomplete, mixed-epoch or incompatible reports.

To append an already completed evaluation without rerunning inference (from tools):

```bash
python boreas_benchmark/export_excel.py \
  --model pv_rcnn --tag seed42_beam128 --epoch 80
```

A script already running when this feature was installed may require that manual
export command once it finishes. To change the shared workbook location:

```bash
BOREAS_EXCEL_PATH=/home/ducanh/projects/openPCDet/output/boreas_summary/my_results.xlsx \
  bash boreas_benchmark/run_metrics.sh second seed666_beam128 80 42
```

If only Excel export fails, JSON results are retained: fix the reported problem
and rerun `export_excel.py`, not the GPU evaluation.

CPU regression checks:

```bash
python -m unittest discover -s boreas_benchmark -p 'test_excel.py'
```

## Same scene at 128 / 64 / 32 / 16 rings with detections

After all four test passes, `run_metrics.sh` automatically exports the first
validation frame at the four ring counts, using each pass's own detections.
The images are saved under:

```text
output/boreas_models/<model>/<tag>/eval/epoch_<N>/val/visualizations/<frame_id>/
  comparison_bev.png     # 2x2: 128, 64 / 32, 16; top view
  comparison_3d.png      # same arrangement and common 3D camera
  ring128_bev.png        # separate PNGs for each ring count and view
  ring128_3d.png
  ring64_bev.png
  ring64_3d.png
  ring32_bev.png
  ring32_3d.png
  ring16_bev.png
  ring16_3d.png
  visualization.json    # frame, camera, cutoff, weights and prediction sources
```

Boxes show class and native detection confidence. Green: Car; orange:
Pedestrian; purple: Cyclist. Points are colored by height. White dot: LiDAR
origin; axis markers and a 10 m grid use LiDAR coordinates. BEV is a top view
of the point cloud; 3D uses a common orthographic projection.
All visible prepared points are shown before voxel caps, without random
display thinning. Counts describe the displayed crop and can differ from
the total scan count. Image generation does not rerun inference or need a GPU,
Open3D, a display server, or Matplotlib; it uses NumPy and Pillow.

Select the SAME frame for all four passes when starting the usual metrics run
(from `tools`):

```bash
BOREAS_VIS_FRAME_INDEX=10 BOREAS_VIS_SCORE=0.3 \
  bash boreas_benchmark/run_metrics.sh second seed666_beam128 80 42
# Or BOREAS_VIS_FRAME_ID=<exact ID from ImageSets/val.txt>.
# BOREAS_VIS_FRAME_ID takes precedence over BOREAS_VIS_FRAME_INDEX.
```

Render existing test predictions immediately, without rerunning the metrics:

```bash
python boreas_benchmark/visualize_rings.py \
  --eval-root ../output/boreas_models/second/seed666_beam128/eval/epoch_80/val \
  --frame-index 10 --score-thresh 0.3

# Choose an exact frame and zoom into +/-40 meters for clearer box labels.
python boreas_benchmark/visualize_rings.py \
  --eval-root ../output/boreas_models/second/seed666_beam128/eval/epoch_80/val \
  --frame-id 1598987772079070 --xy-range -40 -40 40 40
```

Defaults: frame index 0, display score >=0.3, full model point-cloud range.
Use `--no-labels` for crowded scenes, `--elevation` / `--azimuth` for the
3D camera, `--data-root` for another prepared dataset parent, and `--out`
for another output parent. The frame ID is appended to the output parent.
Repeated export for the same frame replaces its images.
The display cutoff does not change detections used by AP/calibration metrics.
Boxes with centers outside the selected crop are omitted from the view.

The renderer matches predictions by frame ID, checks identical split IDs,
and verifies prediction hashes and shared checkpoint/inference settings.
It requires `inference_manifest.json` from the current test pipeline.
If visualization fails after test, fix the reported input/dependency issue
and rerun only `visualize_rings.py`. Existing metric results are retained.
To skip automatic images: `BOREAS_VISUALIZE=0 bash boreas_benchmark/run_metrics.sh ...`.

CPU checks from the repository root:

```bash
python -m unittest discover -s tools/boreas_benchmark -p 'test_visualize_rings.py'
```

## Paper point-cloud figures

`paper_sampling_figures.py` renders the same validation frame across the prepared
modulo, elevation-matched/FOV-control, and 16-ring phase datasets. Ground-truth
boxes are used by default, so the sampling illustration is independent of any
detector. The paper preset uses a close \(\pm23\)-m crop and an oblique view.
Each cloud remains on a dark panel with a white border; its short black title is
placed outside the image on the white figure background. No inference or GPU is
required.

From `tools`:

```bash
python boreas_benchmark/paper_sampling_figures.py \
  --suite all --frame-id 1598988267576590 --boxes gt
```

Outputs are written to
`../output/boreas_summary/paper_pointcloud_figures/`:

- `paper_modulo_128_64_32_16_3d.png`: 128/64/32/16-ring comparison.
- `paper_elevation_3d.png`: modulo, target-angle and dense-FOV controls.
- `paper_phase16_3d.png`: all eight 16-ring modulo phases.
- `paper_sampling_figures.json`: frame, crop, camera, point counts, and files.

Matching one-page PDFs are written to `../output/pdf/` with the same stems.

Use `--frame-id ID` to freeze an exact paper frame and `--xy-range` for a
different crop. `--view bev` restores a top view, while `--view both` exports
both views in the same figure. Add `--box-labels` only when class text is
needed. Use `--boxes prediction --eval-root PATH` to show one model's
predictions instead of common ground truth. Every panel shares the same crop,
camera, point rendering, and box source.
## Elevation-angle sensor profiles

New optional mode: `bash boreas_benchmark/run_metrics.sh pointpillar seed666_beam128 80 42 elevation`.
Use `elevation-plan` as argument 5 to inspect the mapping without preparing
validation data or running inference. Commands without argument 5 retain the
original modulo benchmark.

Defaults: VLP-16 and VLP-32C, plus dense controls within each target FOV.
Twenty training scans estimate source ring elevations; a distinct-ring optimal
assignment enforces a 2-degree maximum error. Existing checkpoints and full
validation GT are retained. Angular metrics and summary.csv are stored under
separate elevation paths rather than replacing beam-only reports/Excel rows.

See [ELEVATION_SAMPLING.md](ELEVATION_SAMPLING.md) for sensor sources, actual
mapping errors, unsupported HDL profiles, all six run commands, custom Ouster
metadata profiles, output locations and CPU tests.
