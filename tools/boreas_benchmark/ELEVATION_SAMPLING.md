# Elevation-angle sampling: implementation and execution

This extends run_metrics.sh with an evaluation-only angular-profile experiment.
It uses existing dense checkpoints. No training or ray-casting is performed.

## Sensor research and defaults

The catalog lidar_profiles.json records sorted elevation angles, original target
laser IDs, source URL, pinned commit and original source-file SHA256. Values are
converted from ROS Velodyne vert_correction radians to degrees. Only vertical
angles are used; target sensor distance offsets and other corrections are NOT
applied to Boreas coordinates.

Reference commit: 56fc178d2dad4b6d38c6a69aeb2435ff75503e52,
[ROS Velodyne calibration files](https://github.com/ros-drivers/velodyne/tree/56fc178d2dad4b6d38c6a69aeb2435ff75503e52/velodyne_pointcloud/params).

| Profile | Beams | Vertical limits (degrees) | Default use |
|---|---:|---:|---|
| vlp16 | 16 | -15 to +15 | enabled |
| vlp32c | 32 | -25 to +15 | enabled |
| hdl32e | 32 | -30.67 to +10.67 | catalog only; source FOV cannot cover it |
| hdl64e_s3_xiesc | 64 | -24.555073 to +1.960097 | catalog only; a specific S3 unit, not a universal HDL-64E table |

Ouster OS1 supports multiple hardware revisions and Uniform/Gradient/Above/Below
Horizon configurations. A generic OS1-32 name or overall FOV does not determine
its individual calibrated beams:
[OS1 Rev7 manufacturer documentation](https://docs.ouster.com/sensor-docs/hardware/os1-overview).
Use beam_altitude_angles from the actual sensor metadata in a custom catalog,
rather than inventing a uniform array for a sensor called OS1-32.

## Method

1. Read XYZIRT from the ORIGINAL Boreas .bin files. Prepared XYZ/intensity .npy
   files lack ring IDs and cannot be used to estimate or select source beams.
2. Select 20 training frame IDs, evenly spaced in the saved training list. Use
   ranges >=5 m. Calculate median atan2(z,hypot(x,y)) for each ring per scan,
   then median across scans. This is an empirical estimate, not factory
   calibration. Every selected scan must observe all 128 rings; reject unstable
   angles if inter-scan median spread exceeds 0.1 degree.
3. Restrict eligible source rings to the target vertical interval (endpoint
   tolerance 0.0001 degree for floating-point error).
4. Match sorted target angles to DISTINCT source rings using dynamic programming
   to minimize total absolute angular error. Enforce a maximum error of 2
   degrees for each pair by default. This avoids duplicate nearest-neighbor
   selections while retaining exactly K rings. Never extrapolate missing rays.
5. Retain original XYZ/intensity and original point order. Copy the exact dense
   validation infos, IDs and labels; keep zero-point GT objects.
6. Also prepare a dense-FOV control: all source rings inside the same target
   vertical interval. This measures FOV cropping without K-beam reduction.
   VLP-32C covers the complete source FOV, so its dense-FOV control is equivalent
   to dense128; its evaluation is repeated explicitly.
7. Hash source/calibration scans, prepared scans, target profile, sampling plan
   and GT. Publish prepared datasets only after all frames finish. Reuse only
   matching, unchanged datasets; otherwise choose a new dataset output root.

Angular selection is fixed across frames and models. This approximates the
target angular profile; it does not reproduce its horizontal resolution, timing,
noise, range, intensity response, beam divergence or return policy. No target
ray is synthesized and no point is moved to a target angle.

The comparison to the original modulo result combines angular-layout/FOV
changes. The dense-FOV control isolates cropping, but does not turn that
comparison into an equal-FOV, equal-K layout-only causal experiment. Same ring
count does not imply equal point count. Reports retain actual point counts and
density recall, alongside fixed-dense128 object cohorts.

## Read-only source check performed on 2026-10-10

Twenty training scans across the saved training split were inspected; no detector
inference or full validation dataset generation was run.

| Target | Assigned rings | Mean absolute error | Maximum absolute error |
|---|---:|---:|---:|
| VLP-16 | 16 | 0.438812 degrees | 1.435000 degrees |
| VLP-32C | 32 | 0.077938 degrees | 0.633000 degrees |

Source angles span approximately -25 to +15 degrees. Maximum inter-scan median
spread across 128 rings was 3.14e-7 degrees. The VLP-16 dense-FOV control retains
125 source rings; the VLP-32C control retains 128.

HDL-32E is rejected because its lower beams lie outside source coverage.
HDL-64E S3 cannot assign 64 distinct rings within the 2-degree bound: the source
has too few beams at large downward angles. A diagnostic unconstrained matching
had maximum error 13.413581 degrees; it is not an acceptable faithful
approximation and is not enabled by default. Do not relax the bound simply to
claim a realistic 64-beam sensor.

## Run in WSL using the AI environment

Both repository-root and tools-directory invocation work. The script changes
directory to tools itself. The fourth argument is the EVALUATION seed. The
checkpoint tag remains the actual saved folder name.

First inspect the mapping without preparing validation scans or using the GPU:

```bash
cd /home/ducanh/projects/openPCDet
bash tools/boreas_benchmark/run_metrics.sh pointpillar seed666_beam128 80 42 elevation-plan
```

Run the new conditions for one saved model:

```bash
bash tools/boreas_benchmark/run_metrics.sh pointpillar seed666_beam128 80 42 elevation
```

This prepares validation-only datasets once, checks them, then evaluates VLP-16,
VLP-16 dense-FOV, VLP-32C and VLP-32C dense-FOV. Subsequent models reuse the same
verified datasets. It does not rerun dense profiling, visualization or the
legacy beam-only Excel exporter.

All six existing epoch-80 checkpoints:

```bash
for model in pointpillar second centerpoint; do
  bash tools/boreas_benchmark/run_metrics.sh "$model" seed666_beam128 80 42 elevation
done
for model in voxelnext pv_rcnn pointpillar_lite; do
  bash tools/boreas_benchmark/run_metrics.sh "$model" seed42_beam128 80 42 elevation
done
```

Defaults evaluate four conditions per model (24 evaluation passes for six).
The original command without argument 5 still runs modulo 128/64/32/16 and its
existing profiling, visualization and Excel export.

Optional settings:

```bash
# Just the VLP-16 profile and its FOV control:
BOREAS_ELEVATION_PROFILES="vlp16" bash tools/boreas_benchmark/run_metrics.sh pointpillar seed666_beam128 80 42 elevation

# Only angular profiles, no dense-FOV controls:
BOREAS_FOV_CONTROLS=0 bash tools/boreas_benchmark/run_metrics.sh second seed666_beam128 80 42 elevation

# Explicit Python if the AI environment is not active:
BOREAS_PYTHON=/home/ducanh/miniforge3/envs/AI/bin/python bash tools/boreas_benchmark/run_metrics.sh second seed666_beam128 80 42 elevation-plan
```

Other settings: BOREAS_DATA_ROOT, BOREAS_RAW_ROOT, BOREAS_ELEVATION_DATA_ROOT,
BOREAS_ELEVATION_CATALOG, BOREAS_CALIBRATION_FRAMES, BOREAS_MAX_ANGLE_ERROR,
BOREAS_DENSITY_CACHE. Angular data defaults to boreas_openpcdet/elevation.

A custom catalog uses the same JSON structure as lidar_profiles.json:
version plus a profiles dictionary. Each profile requires name, description,
source_url and angles_deg sorted strictly increasing, with 1..128 distinct
values in degrees. For an Ouster device, sort its actual beam_altitude_angles
and record device model, revision, beam pattern, metadata checksum and source
location in the description/provenance fields. Target angles beyond source
coverage or matching error bounds fail before validation preparation/inference.

## Outputs

Per condition:

```text
output/boreas_models/<model>/<actual-tag>/eval/epoch_80/val/elevation_<condition>/
  result.pkl
  accuracy.json
  range_ap.json
  kitti_accuracy.json
  inference_manifest.json
  sampling_manifest.json
output/boreas_models/<model>/<actual-tag>/elevation/epoch_80/<condition>/
  extended.json
  reliability_<rings>_<class>.svg
output/boreas_models/<model>/<actual-tag>/elevation/epoch_80/summary.csv
```

The CSV contains per-class AP40, mAP40 by range, actual ring/point counts and
angular errors. If existing same-epoch modulo reports have matching GT,
checkpoint and inference settings, it includes dense128 and same-ring-count
modulo mAP plus differences in percentage points. Missing baselines stay blank;
incompatible or stale baselines are rejected.

These outputs are isolated from existing range_ap_16/32/64.json,
extended_16/32/64.json and the shared beam-only workbook. A condition identity
contains both beam count and angular profile, so it must not replace a legacy
16/32/64 Excel row.

## Verification

```bash
bash -n tools/boreas_benchmark/run_metrics.sh
python -m unittest discover -s tools/boreas_benchmark -p 'test_elevation.py' -v
python -m unittest discover -s tools/boreas_benchmark -p 'test_extended.py' -v
```

Tests cover injective optimal matching, FOV/error rejection, original points
and intensity, training-only estimation, unchanged GT, dataset reuse/tampering,
baseline provenance, shell plan mode and an entire shell evaluation pipeline
with a fake detector on one synthetic CPU frame.
