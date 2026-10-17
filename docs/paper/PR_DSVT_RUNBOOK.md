# DSVT and PR-DSVT on Boreas

Run commands from `/home/ducanh/projects/openPCDet/tools` in the Python
environment that contains the compiled OpenPCDet CUDA dependencies.

## Dependency preflight

```bash
python3 -c "import torch, spconv, kornia; print(torch.__version__, torch.cuda.is_available())"
```

## Train seed 42

As in the six existing seed-42 runs, the physical and effective training batch
are both four. `GRAD_ACCUM_STEPS=1` for both configurations.

```bash
cd /home/ducanh/projects/openPCDet/tools

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --cfg_file cfgs/boreas_models/dsvt.yaml \
  --extra_tag seed42 --boreas_beams 128 \
  --fix_random_seed --seed 42 --batch_size 4 --workers 2 --use_amp

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --cfg_file cfgs/boreas_models/pr_dsvt.yaml \
  --extra_tag seed42 --boreas_beams 128 \
  --fix_random_seed --seed 42 --batch_size 4 --workers 2 --use_amp
```

PR-DSVT samples dense/64/32/16 inputs with total probabilities
`.40/.20/.20/.20`; phase probability is uniform within each sparse beam count.
Validation remains deterministic and uses only the explicitly selected data
path.

If physical batch four exceeds GPU memory, use `--batch_size 1` together with
`--set OPTIMIZATION.GRAD_ACCUM_STEPS 4`. This preserves the effective batch but
is a documented fallback rather than the primary benchmark setting.

## Resume an interrupted run

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --cfg_file cfgs/boreas_models/pr_dsvt.yaml \
  --extra_tag seed42 --boreas_beams 128 \
  --fix_random_seed --seed 42 --batch_size 4 --workers 2 --use_amp --resume
```

## Dense checkpoint evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py \
  --cfg_file cfgs/boreas_models/dsvt.yaml \
  --ckpt ../output/boreas_models/dsvt/seed42_beam128/ckpt/checkpoint_epoch_80.pth \
  --extra_tag seed42 --eval_tag beam128 --boreas_beams 128 \
  --batch_size 1 --workers 2 --seed 42 --range_eval

CUDA_VISIBLE_DEVICES=0 python3 test.py \
  --cfg_file cfgs/boreas_models/pr_dsvt.yaml \
  --ckpt ../output/boreas_models/pr_dsvt/seed42_beam128/ckpt/checkpoint_epoch_80.pth \
  --extra_tag seed42 --eval_tag beam128 --boreas_beams 128 \
  --batch_size 1 --workers 2 --seed 42 --range_eval
```

## Complete paper evaluation

This reuses one frozen epoch-80 checkpoint per model and runs modulo,
elevation-profile, and all modulo-phase conditions.

```bash
cd /home/ducanh/projects/openPCDet/tools

CUDA_VISIBLE_DEVICES=0 BOREAS_PYTHON=python3 \
  bash boreas_benchmark/run_metrics.sh dsvt seed42_beam128 80 42 all

CUDA_VISIBLE_DEVICES=0 BOREAS_PYTHON=python3 \
  bash boreas_benchmark/run_metrics.sh pr_dsvt seed42_beam128 80 42 all
```

For only the 2/4/8 phase sweep:

```bash
CUDA_VISIBLE_DEVICES=0 BOREAS_PYTHON=python3 \
  bash boreas_benchmark/run_metrics.sh pr_dsvt seed42_beam128 80 42 multiphase
```

Do not add either model to the manuscript until its run manifest, checkpoint,
prediction hashes, phase summary, and profiling report pass the same audit as
the existing six configurations.
