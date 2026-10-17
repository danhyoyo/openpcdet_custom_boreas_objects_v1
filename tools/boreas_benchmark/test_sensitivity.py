"""CPU regression checks for AP export sensitivity."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extended import sha256_file
from sensitivity import audit, filter_predictions

BOX = np.array([0, 0, 0, 4, 2, 2, 0.], dtype=np.float32)


class SensitivityTests(unittest.TestCase):
    def test_filter_is_score_ordered_stable_and_frame_local(self):
        records = [{"frame_id": "f", "name": np.array(["Car"]*4),
                    "boxes_lidar": np.stack([BOX]*4),
                    "score": np.array([.5, .9, .9, .01])}]
        result = filter_predictions(records, .1, 2)[0]
        self.assertEqual(result["score"].tolist(), [.9, .9])
        self.assertEqual(result["boxes_lidar"].shape, (2, 7))

    def test_audit_rejects_unrecoverable_direction_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            infos = [{"point_cloud": {"lidar_idx": "f"},
                      "annos": {"name": np.array(["Car"]),
                                "gt_boxes_lidar": np.array([BOX])}}]
            records = [{"frame_id": "f", "name": np.array(["Car"]),
                        "boxes_lidar": np.array([BOX]), "score": np.array([.9])}]
            pred, info, manifest = root/"result.pkl", root/"infos.pkl", root/"manifest.json"
            pred.write_bytes(pickle.dumps(records)); info.write_bytes(pickle.dumps(infos))
            manifest.write_text(json.dumps({"score_cutoff": .001, "nms_post_maxsize": 500,
                                            "prediction_sha256": sha256_file(pred)}))
            report = audit(pred, info, manifest, [.001, .1], [1, 500])
            self.assertEqual(len(report["conditions"]), 4)
            self.assertTrue(all(row["mAP40"] == 100 for row in report["conditions"]))
            with self.assertRaises(ValueError):
                audit(pred, info, manifest, [0.], [500])
            with self.assertRaises(ValueError):
                audit(pred, info, manifest, [.001], [501])
            saved = json.loads(manifest.read_text()); saved["prediction_sha256"] = "bad"
            manifest.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                audit(pred, info, manifest, [.001], [500])


if __name__ == "__main__":
    unittest.main()
