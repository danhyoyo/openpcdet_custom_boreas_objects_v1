"""Generate six Boreas model configs from this OpenPCDet checkout's templates."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

CLASSES = ["Car", "Pedestrian", "Cyclist"]
MODEL_SOURCES = {
    "pointpillar": "kitti_models/pointpillar.yaml",
    "second": "custom_models/second.yaml",
    "centerpoint": "once_models/centerpoint.yaml",
    "voxelnext": "nuscenes_models/cbgs_voxel0075_voxelnext.yaml",
    "pv_rcnn": "custom_models/pv_rcnn.yaml",
}


def anchor_statistics(data_path: Path) -> dict:
    info_path = data_path / "custom_infos_train.pkl"
    if not info_path.is_file():
        return {}
    with info_path.open("rb") as handle:
        infos = pickle.load(handle)
    by_class = {name: [] for name in CLASSES}
    for info in infos:
        annotations = info["annos"]
        for name, box in zip(annotations["name"], annotations["gt_boxes_lidar"]):
            if name in by_class:
                by_class[name].append(box)
    result = {}
    for name, boxes in by_class.items():
        if not boxes:
            raise ValueError(f"No training boxes for {name}; inspect the split")
        array = np.asarray(boxes)
        result[name] = {
            "count": len(array),
            "anchor_sizes": np.median(array[:, 3:6], axis=0).round(3).tolist(),
            "anchor_bottom_height": round(float(np.median(array[:, 2] - array[:, 5] / 2)), 3),
        }
    return result


def make_one(name: str, source: Path, statistics: dict) -> dict:
    config = yaml.safe_load(source.read_text())
    config["CLASS_NAMES"] = CLASSES
    config["DATA_CONFIG"] = {"_BASE_CONFIG_": "cfgs/boreas_models/dataset.yaml"}
    config["OPTIMIZATION"]["BATCH_SIZE_PER_GPU"] = 1
    config["OPTIMIZATION"]["NUM_EPOCHS"] = 80
    model = config["MODEL"]
    model["POST_PROCESSING"]["EVAL_METRIC"] = "kitti"

    dense = model["DENSE_HEAD"]
    post = dense.get("POST_PROCESSING", model["POST_PROCESSING"])
    post["SCORE_THRESH"] = 0.001
    post["NMS_CONFIG"]["NMS_PRE_MAXSIZE"] = 4096
    post["NMS_CONFIG"]["NMS_POST_MAXSIZE"] = 500
    if "ANCHOR_GENERATOR_CONFIG" in dense:
        for anchor in dense["ANCHOR_GENERATOR_CONFIG"]:
            if anchor["class_name"] == "Vehicle":
                anchor["class_name"] = "Car"
            if anchor["class_name"] in statistics:
                values = statistics[anchor["class_name"]]
                anchor["anchor_sizes"] = [values["anchor_sizes"]]
                anchor["anchor_bottom_heights"] = [values["anchor_bottom_height"]]
    if "CLASS_NAMES_EACH_HEAD" in dense:
        dense["CLASS_NAMES_EACH_HEAD"] = [CLASSES]
    if "POST_PROCESSING" in dense:
        dense["POST_PROCESSING"]["POST_CENTER_LIMIT_RANGE"] = [
            -75.2, -75.2, -3.0, 75.2, 75.2, 5.0
        ]
    if name == "voxelnext":
        separate = dense["SEPARATE_HEAD_CFG"]
        separate["HEAD_ORDER"] = ["center", "center_z", "dim", "rot"]
        separate["HEAD_DICT"].pop("vel", None)
        dense["LOSS_CONFIG"]["LOSS_WEIGHTS"]["code_weights"] = [1.0] * 8
    if name == "pointpillar":
        config["DATA_CONFIG"]["DATA_PROCESSOR"] = [
            {"NAME": "mask_points_and_boxes_outside_range", "REMOVE_OUTSIDE_BOXES": True},
            {"NAME": "shuffle_points", "SHUFFLE_ENABLED": {"train": True, "test": False}},
            {"NAME": "transform_points_to_voxels", "VOXEL_SIZE": [0.2, 0.2, 8.0],
             "MAX_POINTS_PER_VOXEL": 32,
             "MAX_NUMBER_OF_VOXELS": {"train": 80000, "test": 80000}},
        ]
    return config


def generate(data_path: Path) -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    output = root / "tools/cfgs/boreas_models"
    output.mkdir(parents=True, exist_ok=True)
    dataset = yaml.safe_load((Path(__file__).with_name("dataset.yaml")).read_text())
    dataset["DATA_PATH"] = str(data_path.resolve())
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    statistics = anchor_statistics(data_path)
    if not statistics:
        print("WARNING: training infos missing; using template anchors until prepare.py is run")
    (output / "anchor_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    configs = {}
    for name, relative in MODEL_SOURCES.items():
        source = root / "tools/cfgs" / relative
        configs[name] = make_one(name, source, statistics)
    lite = copy.deepcopy(configs["pointpillar"])
    lite_model = lite["MODEL"]
    lite_model["VFE"]["NUM_FILTERS"] = [32]
    lite_model["MAP_TO_BEV"]["NUM_BEV_FEATURES"] = 32
    lite_model["BACKBONE_2D"]["NUM_FILTERS"] = [32, 64, 128]
    lite_model["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"] = [64, 64, 64]
    configs["pointpillar_lite"] = lite
    paths = []
    for name, config in configs.items():
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path,
                        default=Path("/home/ducanh/datasets/boreas_openpcdet/128"))
    args = parser.parse_args()
    for path in generate(args.data_path):
        print(path)


if __name__ == "__main__":
    main()
