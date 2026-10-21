# OpenPCDet for Boreas

This repository extends **OpenPCDet** with support for the
**Boreas-Objects-V1** dataset. It provides the data conversion utilities,
dataset configuration and model configurations needed to train and evaluate
OpenPCDet detectors on Boreas point clouds.

The core 3D detection framework and detector implementations come from
[OpenPCDet by the OpenPCDet Development Team](https://github.com/open-mmlab/OpenPCDet).
The additions in this repository focus on integrating Boreas into that framework.

## Boreas integration

- Convert labelled Boreas LiDAR scans to OpenPCDet's `CustomDataset` format.
- Generate training and validation metadata from the Boreas object-detection
  split definitions.
- Map Boreas annotations to the Car, Pedestrian and Cyclist training classes.
- Provide Boreas dataset and model configurations for PointPillar, SECOND,
  CenterPoint, VoxelNeXt, PV-RCNN and DSVT.
- Use the standard OpenPCDet workflow for training and evaluation.

Dataset files, trained weights and generated outputs are intentionally excluded
from version control.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`tools/boreas_benchmark/prepare.py`](tools/boreas_benchmark/prepare.py) | Download and convert Boreas-Objects-V1 data |
| [`tools/boreas_benchmark/make_configs.py`](tools/boreas_benchmark/make_configs.py) | Generate Boreas model configurations |
| [`tools/cfgs/boreas_models/`](tools/cfgs/boreas_models/) | Boreas dataset and detector configurations |
| [`pcdet/datasets/custom/`](pcdet/datasets/custom/) | OpenPCDet custom-dataset implementation used by Boreas |
| [`docs/`](docs/) | OpenPCDet installation and usage documentation |

## Installation

Use Linux or WSL2 with an NVIDIA GPU for CUDA training and inference. Follow the
[OpenPCDet installation guide](docs/INSTALL.md) to install compatible versions
of PyTorch, CUDA and spconv, then build the project extensions.

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
pip install asrl-pyboreas boto3
```

`asrl-pyboreas` supplies the Boreas split definitions used during conversion.
`boto3` is required only when using the optional dataset download command.

## Prepare Boreas data

Choose local directories for the downloaded and converted data. The example
below uses the ignored `data/` directory inside the repository.

```bash
export BOREAS_SOURCE_ROOT="$PWD/data/boreas"
export BOREAS_DATA_ROOT="$PWD/data/boreas_openpcdet"

# Inspect the download before transferring the labelled scans.
python tools/boreas_benchmark/prepare.py download \
  --destination "$BOREAS_SOURCE_ROOT" --plan

# Download and convert Boreas-Objects-V1.
python tools/boreas_benchmark/prepare.py download \
  --destination "$BOREAS_SOURCE_ROOT"
python tools/boreas_benchmark/prepare.py prepare \
  --source "$BOREAS_SOURCE_ROOT" \
  --destination "$BOREAS_DATA_ROOT"
```

The converter writes point clouds, labels, split lists, OpenPCDet info files
and a preparation manifest. `Misc` annotations are retained so they can be
handled correctly during evaluation, while the configured training classes are
Car, Pedestrian and Cyclist.

Set `DATA_PATH` in
[`tools/cfgs/boreas_models/dataset.yaml`](tools/cfgs/boreas_models/dataset.yaml)
to the converted dataset directory. You can also regenerate the model configs
for that directory:

```bash
python tools/boreas_benchmark/make_configs.py \
  --data-path /path/to/converted/boreas/data
```

## Train a detector

Run OpenPCDet training from `tools/` after preparing the dataset:

```bash
cd tools
python train.py --cfg_file cfgs/boreas_models/pointpillar.yaml \
  --batch_size 1 --epochs 80 --workers 2 \
  --fix_random_seed --seed 42 \
  --extra_tag boreas_pointpillar
```

Replace the configuration file to train another supported detector. Training
artifacts are written under `output/`, which is excluded from version control.

## Evaluate a checkpoint

Use the standard OpenPCDet evaluation entry point with the same model
configuration used for training:

```bash
python test.py \
  --cfg_file cfgs/boreas_models/pointpillar.yaml \
  --ckpt /path/to/checkpoint.pth \
  --batch_size 1 --workers 2
```

Evaluation outputs are also written under `output/` and remain local.

## Attribution and license

This project is derived from
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and is distributed under
the [Apache License 2.0](LICENSE). Preserve the upstream license and notices
when redistributing this code. Boreas dataset access and use are governed by
the dataset provider's terms.

When using this repository in research, cite OpenPCDet and the detector methods
you use:

```bibtex
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```