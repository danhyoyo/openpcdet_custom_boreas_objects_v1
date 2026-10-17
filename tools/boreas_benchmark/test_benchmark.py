"""CPU regression checks for the local metric and generated inference configs."""
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ap40_from_ranked, ap_for_class, evaluate, iou3d, in_range
from prepare import beam_subset
from make_configs import make_one, MODEL_SOURCES
from report import write_reports

BOX = np.array([10, 0, 0, 4, 2, 2, 0], dtype=np.float32)
def record(names, boxes, scores=None):
    value = {"name": np.array(names), "gt_boxes_lidar": np.array(boxes, dtype=np.float32).reshape(-1, 7)}
    if scores is not None:
        value.update(boxes_lidar=value.pop("gt_boxes_lidar"), score=np.array(scores), frame_id="f")
    return value

class MetricTests(unittest.TestCase):
    def score(self, pred, truth=None, low=0, high=float("inf")):
        truth = record(["Car"], [BOX]) if truth is None else truth
        return ap_for_class({"f": pred}, {"f": truth}, "Car", .7, low, high)

    def test_geometry(self):
        self.assertAlmostEqual(iou3d(BOX, BOX), 1)
        shifted = BOX.copy(); shifted[0] += 1
        self.assertAlmostEqual(iou3d(BOX, shifted), .6, places=6)
        rotated = BOX.copy(); rotated[6] = np.pi / 2
        self.assertAlmostEqual(iou3d(BOX, rotated), 1/3, places=6)
        far = BOX.copy(); far[2] += 3
        self.assertEqual(iou3d(BOX, far), 0)

    @staticmethod
    def _corners_reference(box):
        c, s = np.cos(box[6]), np.sin(box[6])
        return [(box[0] + c*sx - s*sy, box[1] + s*sx + c*sy)
                for sx, sy in ((box[3]/2, box[4]/2), (-box[3]/2, box[4]/2),
                               (-box[3]/2, -box[4]/2), (box[3]/2, -box[4]/2))]

    def test_geometry_against_independent_shapely_reference(self):
        try:
            from shapely.geometry import Polygon
        except ImportError:
            self.skipTest("Shapely is optional; analytical IoU tests still run")
        cases = []
        for x, y, z, yaw in ((10, 0, 0, .37), (10.7, -.4, .25, -1.1),
                             (8.5, 1.2, -.2, 2.4)):
            box = BOX.astype(float).copy()
            box[:3], box[6] = (x, y, z), yaw
            cases.append(box)
        for first in cases:
            for second in cases:
                area = Polygon(self._corners_reference(first)).intersection(
                    Polygon(self._corners_reference(second))).area
                z_overlap = max(0., min(first[2]+first[5]/2, second[2]+second[5]/2) -
                                max(first[2]-first[5]/2, second[2]-second[5]/2))
                overlap = area*z_overlap
                union = np.prod(first[3:6])+np.prod(second[3:6])-overlap
                self.assertAlmostEqual(iou3d(first, second), overlap/union if overlap else 0., places=9)

    def test_precision_envelope_and_recall_sampling(self):
        self.assertEqual(ap40_from_ranked([], [], 1), 0)
        self.assertIsNone(ap40_from_ranked([], [], 0))
        self.assertAlmostEqual(ap40_from_ranked([1, 0, 1], [0, 1, 0], 2), 83.3333, places=4)
        with self.assertRaises(ValueError):
            ap40_from_ranked([1], [1], 1)

    def test_perfect_missing_and_high_score_fp(self):
        self.assertEqual(self.score(record(["Car"], [BOX], [.9]))["AP40"], 100)
        missing = self.score(record([], [], []))
        self.assertEqual(missing["AP40"], 0)
        self.assertEqual(missing["operating_point"]["FN"], 1)
        far = BOX.copy(); far[0] += 20
        self.assertEqual(self.score(record(["Car", "Car"], [far, BOX], [.99, .9]))["AP40"], 50)

    def test_duplicate_and_misc(self):
        dup = self.score(record(["Car", "Car"], [BOX, BOX], [.9, .8]))
        self.assertEqual(dup["operating_point"]["TP"], 1)
        self.assertEqual(dup["operating_point"]["FP"], 1)
        other = BOX.copy(); other[0] += 20
        truth = record(["Car", "Misc"], [BOX, other])
        ignored = self.score(record(["Car", "Car"], [other, BOX], [.99, .9]), truth)
        self.assertEqual(ignored["AP40"], 100)
        self.assertEqual(ignored["ignored_misc"], 1)
        # A match to a target takes priority over overlapping Misc.
        truth = record(["Car", "Misc"], [BOX, BOX])
        self.assertEqual(self.score(record(["Car"], [BOX], [.9]), truth)["operating_point"]["TP"], 1)

    def test_range_boundary_empty_class_and_low_score(self):
        edge = BOX.copy(); edge[0] = 20
        self.assertFalse(in_range(edge, 0, 20))
        self.assertTrue(in_range(edge, 20, 40))
        self.assertIsNone(self.score(record([], [], []), record([], []))["AP40"])
        low = self.score(record(["Car"], [BOX], [.01]))
        self.assertEqual(low["AP40"], 100)
        self.assertEqual(low["operating_point"]["TP"], 0)
        self.assertEqual(low["operating_point"]["FN"], 1)

    def test_report_protocol_and_frame_validation(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            infos = [{"point_cloud": {"lidar_idx": "f"}, "annos": record(["Car"], [BOX])}]
            (folder / "custom_infos_val.pkl").write_bytes(pickle.dumps(infos))
            (folder / "result.pkl").write_bytes(pickle.dumps([record(["Car"], [BOX], [.9])]))
            dataset = SimpleNamespace(root_path=folder, split="val", class_names=["Car", "Pedestrian", "Cyclist"])
            write_reports(dataset, folder, {"Car_3d/moderate_R40": 99.}, True, Mock())
            primary = json.loads((folder / "accuracy.json").read_text())
            self.assertEqual(primary["classes"]["Car"], 100)
            self.assertEqual(primary["protocol_version"], "local_ap40_v1")
            self.assertTrue((folder / "kitti_accuracy.json").exists())
            write_reports(dataset, folder, {}, False, Mock())
            self.assertIsNone(json.loads((folder / "accuracy.json").read_text())["mAP40"])
            (folder / "result.pkl").write_bytes(pickle.dumps([]))
            with self.assertRaises(ValueError):
                evaluate(folder / "result.pkl", folder / "custom_infos_val.pkl")

    def test_modulo_phase_subsets_are_complete_and_distinct(self):
        points = np.zeros((128, 6), dtype=np.float32)
        points[:, 4] = np.arange(128)
        for beams in (64, 32, 16):
            modulus = 128 // beams
            retained = []
            for phase in range(modulus):
                subset = beam_subset(points, beams, phase)
                self.assertEqual(len(subset), beams)
                self.assertTrue(np.all(subset[:, 4].astype(int) % modulus == phase))
                retained.extend(subset[:, 4].astype(int).tolist())
            self.assertEqual(sorted(retained), list(range(128)))
        with self.assertRaises(ValueError):
            beam_subset(points, 64, 2)

class ConfigTests(unittest.TestCase):
    def test_generated_and_saved_configs_have_common_inference_budget(self):
        cfg_root = Path(__file__).resolve().parents[1] / "cfgs"
        for name, source in MODEL_SOURCES.items():
            for cfg in (make_one(name, cfg_root / source, {}), yaml.safe_load((cfg_root / "boreas_models" / f"{name}.yaml").read_text())):
                post = cfg["MODEL"]["DENSE_HEAD"].get("POST_PROCESSING", cfg["MODEL"]["POST_PROCESSING"])
                self.assertEqual(post["SCORE_THRESH"], .001, name)
                self.assertEqual(post["NMS_CONFIG"]["NMS_PRE_MAXSIZE"], 4096, name)
                self.assertEqual(post["NMS_CONFIG"]["NMS_POST_MAXSIZE"], 500, name)
        cfg = yaml.safe_load((cfg_root / "boreas_models/pointpillar_lite.yaml").read_text())
        self.assertEqual(cfg["MODEL"]["POST_PROCESSING"]["SCORE_THRESH"], .001)


class TrainingControlTests(unittest.TestCase):
    @staticmethod
    def functions(path, names, namespace):
        import ast
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

    def test_seed_precision_and_resume_arguments(self):
        import argparse
        from types import ModuleType
        from unittest.mock import patch
        class Config(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        def parse(argv, config_amp=False):
            cfg = Config(OPTIMIZATION=Config(USE_AMP=config_amp))
            ns = dict(argparse=argparse, Path=Path, cfg=cfg,
                      cfg_from_yaml_file=lambda *args: None,
                      cfg_from_list=lambda *args: cfg.OPTIMIZATION.update(USE_AMP=True))
            self.functions(Path(__file__).resolve().parents[1] / "train.py", {"parse_config"}, ns)
            beams = ModuleType("boreas_benchmark.beam_config")
            beams.select_beams = lambda *args: None
            with patch.dict(sys.modules, {"boreas_benchmark.beam_config": beams}), patch.object(sys, "argv", ["train.py", "--cfg_file", "cfgs/example.yaml"] + argv):
                return ns["parse_config"]()[0]
        self.assertEqual(parse([]).seed, 42)
        self.assertEqual(parse(["--seed", "666", "--resume"]).seed, 666)
        self.assertTrue(parse(["--use_amp"]).use_amp)
        self.assertFalse(parse(["--no_amp"], True).use_amp)
        self.assertTrue(parse(["--set", "unused"]).use_amp)
        self.assertFalse(parse(["--no_amp", "--set", "unused"]).use_amp)

    def test_missing_checkpoint_does_not_wait_forever(self):
        import glob, os, re
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            sleeper = Mock()
            ns = dict(glob=glob, os=os, re=re, cfg=SimpleNamespace(LOCAL_RANK=1,
                      DATA_CONFIG=SimpleNamespace(DATA_SPLIT={"test": "val"})),
                      time=SimpleNamespace(sleep=sleeper))
            self.functions(Path(__file__).resolve().parents[1] / "test.py",
                           {"get_no_evaluated_ckpt", "repeat_eval_ckpt"}, ns)
            args = SimpleNamespace(start_epoch=80, max_waiting_mins=0)
            ns["repeat_eval_ckpt"](None, None, args, folder, Mock(), folder)
            sleeper.assert_not_called()

if __name__ == "__main__":
    unittest.main()
