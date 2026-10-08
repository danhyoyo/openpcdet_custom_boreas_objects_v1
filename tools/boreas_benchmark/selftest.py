"""Small data-conversion and AP regression test; uses no Boreas download or GPU."""

from __future__ import annotations

import argparse
import pickle
import tempfile
from importlib import metadata, util
from pathlib import Path

import numpy as np

from evaluate import evaluate, iou3d
from prepare import prepare


def main() -> None:
    spec = util.spec_from_file_location("official_splits", metadata.distribution(
        "asrl-pyboreas").locate_file("pyboreas/data/splits.py"))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "raw/boreas-objects-v1"
        (source / "lidar").mkdir(parents=True)
        (source / "labels_detection").mkdir()
        for index in (0, 1):
            timestamp = str(module.obj_train[index][1] + 1)
            points = np.array([[10, 0, 0, 1, 0, 0],
                               [10, 1, 0, 1, 1, 0],
                               [10, -1, 0, 1, 127, 0]], np.float32)
            points.tofile(source / "lidar" / (timestamp + ".bin"))
            (source / "labels_detection" / (timestamp + ".txt")).write_text(
                "id Car 4 2 2 10 0 0 0 3\n")
        prepare(root / "raw", root / "prepared", 64, 0)
        output = root / "prepared/64"
        with (output / "custom_infos_val.pkl").open("rb") as handle:
            val = pickle.load(handle)
        with (output / "custom_infos_train.pkl").open("rb") as handle:
            train = pickle.load(handle)
        assert len(val) == len(train) == 1
        for info in val + train:
            array = np.load(output / "points" / (info["point_cloud"]["lidar_idx"] + ".npy"))
            assert array.shape == (1, 4), array.shape
        assert iou3d(val[0]["annos"]["gt_boxes_lidar"][0],
                     val[0]["annos"]["gt_boxes_lidar"][0]) == 1.0
        frame = val[0]["point_cloud"]["lidar_idx"]
        prediction = [{"frame_id": frame,
                       "name": np.array(["Car"]),
                       "score": np.array([0.99], np.float32),
                       "boxes_lidar": val[0]["annos"]["gt_boxes_lidar"]}]
        prediction_path = root / "predictions.pkl"
        with prediction_path.open("wb") as handle:
            pickle.dump(prediction, handle)
        metrics = evaluate(prediction_path, output / "custom_infos_val.pkl")
        assert metrics["ranges"]["all"]["classes"]["Car"]["AP40"] == 100.0
        print("Boreas conversion, 64-beam selection, and AP40 smoke test passed")


if __name__ == "__main__":
    main()
