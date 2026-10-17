"""CPU checks for correct same-frame geometry, filtering and PNG export."""
import argparse
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("visualize_rings", Path(__file__).with_name("visualize_rings.py"))
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


class RingVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.evals = self.root / "eval"
        for beam in visual.BEAMS:
            data = self.data / str(beam)
            (data / "ImageSets").mkdir(parents=True)
            (data / "ImageSets" / "val.txt").write_text("a\nb\n")
            (data / "points").mkdir()
            np.save(data / "points" / "b.npy", [[0., 0., 0., 1.], [100., 0., 0., 1.]])
            directory = self.evals / f"beam{beam}"
            directory.mkdir(parents=True)
            # Deliberately reverse prediction order: match by ID, never array index.
            records = [{"frame_id": "b", "boxes_lidar": np.array([[2., 0, 0, 4, 2, 2, 0], [3., 0, 0, 4, 2, 2, 0]]),
                        "name": np.array(["Car", "Cyclist"]), "score": np.array([.9, .1])},
                       {"frame_id": "a", "boxes_lidar": np.zeros((0, 7)),
                        "name": np.array([]), "score": np.array([])}]
            with (directory / "result.pkl").open("wb") as stream:
                pickle.dump(records, stream)
            manifest = {"checkpoint_sha256": "same-weights", "model_config_sha256": "same-model",
                        "precision": "FP32", "point_cloud_range": [-10, -10, -3, 10, 10, 5],
                        "score_cutoff": .001, "data_path": str(data),
                        "prediction_sha256": visual.sha256(directory / "result.pkl")}
            (directory / "inference_manifest.json").write_text(json.dumps(manifest))

    def load(self):
        return visual.load_scenes(self.evals, self.data, "val", None, 1, .3)

    def test_frame_id_not_prediction_position_and_filtering(self):
        frame, scenes, _, _ = self.load()
        self.assertEqual(frame, "b")
        for scene in scenes:
            self.assertEqual(scene["names"].tolist(), ["Car"])
            self.assertEqual(len(scene["points"]), 1)
            self.assertEqual(scene["input_points"], 2)

    def test_rejects_split_mismatch(self):
        (self.data / "16/ImageSets/val.txt").write_text("b\na\n")
        with self.assertRaisesRegex(ValueError, "frame IDs differ"):
            self.load()

    def test_rejects_different_checkpoint(self):
        path = self.evals / "beam32/inference_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["checkpoint_sha256"] = "other-weights"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Weights"):
            self.load()

    def test_rejects_changed_prediction(self):
        with (self.evals / "beam64/result.pkl").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load()

    def test_rejects_missing_frame(self):
        with self.assertRaisesRegex(ValueError, "not in"):
            visual.choose_frame(self.data, "val", "unknown")
        with self.assertRaisesRegex(ValueError, "frame-index"):
            visual.choose_frame(self.data, "val", frame_index=-1)

    def test_rotated_box_uses_lidar_center_and_heading(self):
        corners = visual.box_corners(np.array([10, 20, 3, 4, 2, 2, np.pi / 2]))
        np.testing.assert_allclose(corners.mean(axis=0), [10, 20, 3])
        np.testing.assert_allclose(np.ptp(corners, axis=0), [2, 4, 2])

    def test_png_export_with_empty_detections(self):
        args = argparse.Namespace(eval_root=self.evals, data_root=self.data, split="val",
                                  frame_id=None, frame_index=1, score_thresh=.95,
                                  xy_range=None, out=self.root / "images", elevation=35,
                                  azimuth=-55, no_labels=False)
        output = visual.export(args)
        self.assertEqual(len(list(output.glob("*.png"))), 10)
        with Image.open(output / "comparison_bev.png") as image:
            self.assertEqual(image.size, (2400, 2100))
        metadata = json.loads((output / "visualization.json").read_text())
        self.assertEqual(metadata["frame_id"], "b")
        self.assertTrue(all(row["detections"] == 0 for row in metadata["sources"]))


if __name__ == "__main__":
    unittest.main()
