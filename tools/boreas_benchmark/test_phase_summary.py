"""CPU regression checks for modulo phase aggregation."""
import json
import tempfile
import unittest
from pathlib import Path

from phase_summary import summarize


class PhaseSummaryTests(unittest.TestCase):
    def test_complete_matrix_and_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); eval_root = root/"eval"; data_root = root/"data"
            for beams in (64, 32, 16):
                for phase in range(128//beams):
                    condition = f"beam{beams}" if phase == 0 else f"beam{beams}_phase{phase}"
                    folder = eval_root/condition; folder.mkdir(parents=True)
                    data = data_root/(str(beams) if phase == 0 else f"{beams}_phase{phase}")
                    data.mkdir(parents=True)
                    (data/"manifest.json").write_text(json.dumps({
                        "sampling_strategy":"ring_id_modulo", "beam_count":beams, "phase":phase,
                        "retained_ring_ids":list(range(phase, 128, 128//beams))}))
                    classes = {name:{"AP40": float(beams+phase)}
                               for name in ("Car","Pedestrian","Cyclist")}
                    (folder/"range_ap.json").write_text(json.dumps({
                        "protocol_version":"local_ap40_v1",
                        "prediction_sha256":"prediction", "truth_sha256":"truth",
                        "ranges":{"all":{"mAP40":float(beams+phase),"classes":classes}}}))
                    (folder/"inference_manifest.json").write_text(json.dumps({
                        "checkpoint_sha256":"same", "model_config_sha256":"same",
                        "prediction_sha256":"prediction",
                        "score_cutoff":.001, "nms_pre_maxsize":4096,
                        "nms_post_maxsize":500, "precision":"FP32",
                        "tf32_matmul":False, "tf32_cudnn":True}))
            report = summarize(eval_root, data_root)
            self.assertTrue(report["complete"])
            entry = next(row for row in report["aggregates"]
                         if row["beams"] == 64 and row["metric"] == "mAP40")
            self.assertEqual(entry["phases"], 2)
            self.assertEqual(entry["mean"], 64.5)
            bad = eval_root/"beam16_phase7"/"inference_manifest.json"
            value = json.loads(bad.read_text()); value["checkpoint_sha256"] = "different"
            bad.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                summarize(eval_root, data_root)


if __name__ == "__main__":
    unittest.main()
