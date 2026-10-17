"""CPU tests for angle assignment, provenance and shell dispatch; no GPU."""
import contextlib
import csv
import io
import itertools
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elevation import (unique_match, make_plan, load_profile, select_points,
                       estimate_source, write_dataset, check_dataset, summarize,
                       sha256_file, truth_fingerprint)


def fixture(root):
    raw, reference = root/"raw", root/"128"
    (raw/"lidar").mkdir(parents=True)
    for name in ("ImageSets", "labels", "points"):
        (reference/name).mkdir(parents=True)
    angles = np.linspace(-25, 15, 128)
    # Source ring ordering intentionally differs from increasing angles.
    angles = np.roll(angles, 37)
    points = np.zeros((128, 6), dtype=np.float32)
    points[:, 0] = 10*np.cos(np.radians(angles))
    points[:, 2] = 10*np.sin(np.radians(angles))
    points[:, 3] = np.arange(128)/128
    points[:, 4] = np.arange(128)
    points[:, 5] = .001
    for frame in ("1", "2", "3"):
        points.tofile(raw/"lidar"/(frame+".bin"))
    np.save(reference/"points/3.npy", points[:,:4])
    (reference/"ImageSets/train.txt").write_text("1\n2\n")
    (reference/"ImageSets/val.txt").write_text("3\n")
    (reference/"labels/3.txt").write_text("10 0 0 1 1 1 0 Pedestrian\n")
    infos = [{"point_cloud": {"num_features": 4, "lidar_idx": "3"},
              "annos": {"name": np.array(["Pedestrian"]),
                        "gt_boxes_lidar": np.array([[10,0,0,1,1,1,0]], dtype=np.float32)}}]
    (reference/"custom_infos_val.pkl").write_bytes(pickle.dumps(infos))
    (reference/"manifest.json").write_text(json.dumps({"beam_count":128, "smoke_test_only":False}))
    profile = {"name":"test_profile", "description":"synthetic test profile",
               "source_url":"test://fixture", "angles_deg":[-10., 0., 10.]}
    return raw, reference, points, infos, profile


class MatchingTests(unittest.TestCase):
    def test_assignment_is_unique_and_globally_optimal(self):
        source = np.array([5., 0., 1., 2.])
        target = np.array([.9, 1.1, 4.9])
        selected = unique_match(source, target, 10.)
        self.assertEqual(len(set(selected)), len(target))
        actual = np.abs(source[selected]-target).sum()
        costs = [np.abs(np.sort(source[list(c)])-target).sum()
                 for c in itertools.combinations(range(len(source)), len(target))]
        self.assertAlmostEqual(actual, min(costs))

    def test_independent_nearest_collision_and_error_bound(self):
        source = np.array([0., 1., 2.])
        target = np.array([.9, 1.1])
        self.assertEqual(len(set(np.argmin(abs(source[:,None]-target), axis=0))), 1)
        selected = unique_match(source, target, 1.)
        self.assertEqual(len(set(selected)), 2)
        self.assertLessEqual(np.max(abs(source[selected]-target)), 1.)
        with self.assertRaises(ValueError):
            unique_match(source, target, .2)

    def test_crop_and_unsupported_fov(self):
        c = {"angles_deg":[-16., -14., 0., 14., 16.]}
        p = {"name":"test", "angles_deg":[-15., 0., 15.]}
        plan = make_plan(p, c, 2.)
        self.assertEqual(plan["selected_source_rings"], [1,2,3])
        self.assertEqual(plan["fov_source_rings"], [1,2,3])
        with self.assertRaises(ValueError):
            make_plan({"name":"outside", "angles_deg":[-17., 0.]}, c)

    def test_original_points_and_intensity_preserved(self):
        p = np.arange(24, dtype=np.float32).reshape(4,6)
        p[:,4] = [7,1,7,3]
        np.testing.assert_array_equal(select_points(p,[7,3]), p[[0,2,3],:4])
        p[0,4] = .4
        with self.assertRaises(ValueError):
            select_points(p,[7])

    def test_catalog(self):
        self.assertEqual(len(load_profile("vlp16")["angles_deg"]),16)
        self.assertEqual(len(load_profile("vlp32c")["angles_deg"]),32)
        self.assertEqual(len(load_profile("hdl64e_s3_xiesc")["angles_deg"]),64)


class DatasetTests(unittest.TestCase):
    def test_training_only_angles_atomic_prepare_reuse_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw, ref, points, infos, profile = fixture(root)
            # Validation has deliberately different geometry, not used to fit angles.
            altered = points.copy(); altered[:,2] += 2
            altered.tofile(raw/"lidar/3.bin")
            calibration = estimate_source(raw,ref,2)
            self.assertEqual([x["frame"] for x in calibration["scans"]],["1","2"])
            plan = make_plan(profile,calibration)
            with contextlib.redirect_stdout(io.StringIO()):
                out = write_dataset(raw,ref,root/"prepared",plan)
                same = write_dataset(raw,ref,root/"prepared",plan)
                control = write_dataset(raw,ref,root/"prepared",plan,True)
            self.assertEqual(out,same)
            manifest = check_dataset(out,ref)
            self.assertEqual(manifest["beam_count"],3)
            self.assertEqual(manifest["truth_sha256"],truth_fingerprint(infos))
            self.assertEqual((out/"custom_infos_val.pkl").read_bytes(),(ref/"custom_infos_val.pkl").read_bytes())
            np.testing.assert_array_equal(np.load(out/"points/3.npy"),
                                          select_points(altered,plan["selected_source_rings"]))
            self.assertTrue(check_dataset(control,ref)["beam_count"] > 3)
            self.assertFalse((out/"custom_infos_train.pkl").exists())
            changed = dict(plan); changed["max_allowed_error_deg"] = 3.
            with self.assertRaises(ValueError):
                write_dataset(raw,ref,root/"prepared",changed)
            altered[0,0] += .1; altered.tofile(raw/"lidar/3.bin")
            with self.assertRaisesRegex(ValueError,"Raw validation scan changed"):
                check_dataset(out,ref)

    def test_prepared_point_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw,ref,points,infos,profile=fixture(root)
            plan=make_plan(profile,estimate_source(raw,ref,2))
            with contextlib.redirect_stdout(io.StringIO()):
                out=write_dataset(raw,ref,root/"prepared",plan)
            np.save(out/"points/3.npy",np.zeros((1,4)))
            with self.assertRaisesRegex(ValueError,"Changed prepared scan"):
                check_dataset(out,ref)

    def test_training_validation_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            raw,ref,*_=fixture(Path(d))
            (ref/"ImageSets/train.txt").write_text("1\n3\n")
            with self.assertRaises(ValueError):
                estimate_source(raw,ref,2)

    def test_shell_plan_needs_no_checkpoint_or_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw,ref,points,infos,profile=fixture(root)
            catalog=root/"catalog.json"
            catalog.write_text(json.dumps({"profiles":{"test_profile":profile}}))
            env=dict(os.environ, BOREAS_PYTHON=sys.executable,
                     BOREAS_DATA_ROOT=str(root), BOREAS_RAW_ROOT=str(raw),
                     BOREAS_ELEVATION_CATALOG=str(catalog),
                     BOREAS_ELEVATION_PROFILES="test_profile", BOREAS_CALIBRATION_FRAMES="2")
            script=Path(__file__).with_name("run_metrics.sh")
            proc=subprocess.run(["bash",str(script),"missing_model","missing_tag","80","42","elevation-plan"],
                                cwd="/tmp",env=env,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertIn("selected_source_rings",proc.stdout)
            self.assertFalse((root/"elevation").exists())

    def test_shell_elevation_pipeline_with_cpu_fake_detector(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory(
                prefix="elevation_cpu_test_", dir=repo/"output/boreas_models/pointpillar") as model_dir:
            root=Path(d); raw,ref,points,infos,profile=fixture(root)
            catalog=root/"catalog.json"
            catalog.write_text(json.dumps({"profiles":{"test_profile":profile}}))
            base=Path(model_dir); (base/"ckpt").mkdir()
            (base/"ckpt/checkpoint_epoch_80.pth").write_bytes(b"CPU fixture")
            mock=root/"mock_detector.py"
            mock.write_text("""import json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
from elevation import sha256_file,truth_fingerprint
from evaluate import evaluate
a=sys.argv[2:]
def arg(k): return a[a.index(k)+1]
data=Path(arg('DATA_CONFIG.DATA_PATH')); ckpt=Path(arg('--ckpt'))
folder=ckpt.parent.parent/'eval/epoch_80/val'/arg('--eval_tag')
folder.mkdir(parents=True)
infos=pickle.loads((data/'custom_infos_val.pkl').read_bytes())
records=[dict(frame_id=i['point_cloud']['lidar_idx'],name=np.array([]),
score=np.array([]),boxes_lidar=np.empty((0,7))) for i in infos]
(folder/'result.pkl').write_bytes(pickle.dumps(records))
infer=dict(prediction_sha256=sha256_file(folder/'result.pkl'),
checkpoint_sha256=sha256_file(ckpt),data_path=str(data.resolve()),
evaluation_seed=int(arg('--seed')),precision='FP32')
(folder/'inference_manifest.json').write_text(json.dumps(infer))
report=evaluate(folder/'result.pkl',data/'custom_infos_val.pkl')
report.update(prediction_sha256=infer['prediction_sha256'],
truth_sha256=truth_fingerprint(infos))
(folder/'range_ap.json').write_text(json.dumps(report))
""")
            wrapper=root/"python_wrapper"
            wrapper.write_text("#!/usr/bin/env bash\n"
                + "if [[ \"$1\" == test.py ]]; then\n"
                + f"  shift; exec '{sys.executable}' '{mock}' '{repo/'tools/boreas_benchmark'}' \"$@\"\n"
                + "fi\n"
                + f"exec '{sys.executable}' \"$@\"\n")
            wrapper.chmod(0o755)
            env=dict(os.environ, BOREAS_PYTHON=str(wrapper), BOREAS_DENSITY_CACHE=str(root/"cache"),
                     BOREAS_DATA_ROOT=str(root), BOREAS_RAW_ROOT=str(raw),
                     BOREAS_ELEVATION_CATALOG=str(catalog),
                     BOREAS_ELEVATION_PROFILES="test_profile", BOREAS_CALIBRATION_FRAMES="2")
            proc=subprocess.run(["bash",str(repo/"tools/boreas_benchmark/run_metrics.sh"),
                                "pointpillar",base.name,"80","42","elevation"],
                                cwd="/tmp",env=env,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            for condition in ("test_profile","test_profile_dense_fov"):
                result=base/"eval/epoch_80/val"/("elevation_"+condition)
                self.assertTrue((result/"sampling_manifest.json").exists())
                report=base/"elevation/epoch_80"/condition/"extended.json"
                self.assertTrue(json.loads(report.read_text())["verified_inference"])
            self.assertTrue((base/"elevation/epoch_80/summary.csv").exists())
            self.assertFalse((base/"profile.json").exists())
            self.assertFalse((base/"eval/epoch_80/val/beam128").exists())

    def test_summary_provenance_and_modulo_delta(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw,ref,points,infos,profile=fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                data=write_dataset(raw,ref,root/"prepared",
                                   make_plan(profile,estimate_source(raw,ref,2)))
            base=root/"model"; evalroot=base/"eval/epoch_80/val"
            def result(folder,ap):
                folder.mkdir(parents=True)
                (folder/"result.pkl").write_bytes(pickle.dumps([]))
                infer={"prediction_sha256":sha256_file(folder/"result.pkl"),
                       "checkpoint_sha256":"checkpoint", "data_path":str(data),
                       "evaluation_seed":42,"precision":"FP32"}
                (folder/"inference_manifest.json").write_text(json.dumps(infer))
                report={"prediction_sha256":infer["prediction_sha256"],
                        "truth_sha256":truth_fingerprint(infos),
                        "ranges":{"all":{"mAP40":ap,"classes":{"Car":{"AP40":ap}}}}}
                (folder/"range_ap.json").write_text(json.dumps(report))
            folder=evalroot/"elevation_test_profile"
            result(folder,20)
            (folder/"sampling_manifest.json").write_bytes((data/"manifest.json").read_bytes())
            result(evalroot/"beam128",50)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(base,80)
            with (base/"elevation/epoch_80/summary.csv").open() as stream:
                row=next(csv.DictReader(stream))
            self.assertEqual(float(row["delta_vs_dense128_pp"]),-30.)
            p=evalroot/"beam128/inference_manifest.json"
            value=json.loads(p.read_text()); value["checkpoint_sha256"]="other"
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"Incompatible modulo baseline"):
                summarize(base,80)


if __name__ == "__main__":
    unittest.main()
