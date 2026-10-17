"""Evaluation-only angular-profile subsampling of raw Boreas XYZIRT scans.

No ray synthesis. Source angles estimated ONLY from training scans.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path
import numpy as np
try:
    from .extended import sha256_file, truth_fingerprint
except ImportError:
    from extended import sha256_file, truth_fingerprint

VERSION = "elevation_subsampling_v1"
CATALOG = Path(__file__).with_name("lidar_profiles.json")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ring_ids(points):
    rings = points[:, 4]
    rounded = np.rint(rings)
    if (not np.isfinite(rings).all() or np.any(np.abs(rings-rounded) >= .01)
            or np.any((rounded < 0) | (rounded > 127))):
        raise ValueError("Expected integer source ring IDs in [0,127]")
    return rounded.astype(np.int16)


def read_raw(path):
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 6:
        raise ValueError(f"Expected nonempty XYZIRT scan: {path}")
    points = values.reshape(-1, 6)
    ring_ids(points)
    if not np.isfinite(points[:, :4]).all():
        raise ValueError(f"Nonfinite XYZ/intensity: {path}")
    return points


def load_profile(name, catalog=CATALOG):
    profiles = json.loads(Path(catalog).read_text())["profiles"]
    if name not in profiles:
        raise ValueError(f"Unknown profile {name}; available: {', '.join(profiles)}")
    profile = profiles[name]
    if profile.get("name") != name or not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError("Profile names must be safe directory names matching catalog keys")
    angles = np.asarray(profile["angles_deg"], dtype=float)
    if (angles.ndim != 1 or not 1 <= len(angles) <= 128 or not np.isfinite(angles).all()
            or np.any(np.diff(angles) <= 0) or np.any(np.abs(angles) >= 90)):
        raise ValueError("Target angles must be distinct, increasing degrees in (-90,90)")
    if not profile.get("source_url") or not profile.get("description"):
        raise ValueError("Custom profiles need source_url and description")
    return profile


def estimate_source(raw, reference, frames=20, min_range=5., max_spread=.1):
    train_path = reference / "ImageSets/train.txt"
    train = train_path.read_text().split()
    val = (reference / "ImageSets/val.txt").read_text().split()
    if len(train) < 2 or set(train) & set(val) or len(train) != len(set(train)):
        raise ValueError("Calibration training frames must be unique and disjoint from validation")
    if frames < 2 or min_range <= 0 or max_spread <= 0:
        raise ValueError("Need >=2 calibration frames and positive range/spread limits")
    indices = np.unique(np.linspace(0, len(train)-1, min(frames, len(train)), dtype=int))
    samples, records = [], []
    for index in indices:
        frame = train[index]
        path = raw / "lidar" / (frame + ".bin")
        points = read_raw(path)
        rings = ring_ids(points)
        xyz = points[:, :3].astype(np.float64)
        valid = np.linalg.norm(xyz, axis=1) >= min_range
        theta = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
        medians = [float(np.median(theta[valid & (rings == ring)]))
                   if np.any(valid & (rings == ring)) else np.nan for ring in range(128)]
        samples.append(medians)
        records.append({"frame": frame, "scan_sha256": sha256_file(path)})
    values = np.asarray(samples)
    if not np.isfinite(values).all():
        raise ValueError("Each calibration scan must observe all 128 rings above min range")
    angles, spread = np.median(values, axis=0), np.ptp(values, axis=0)
    if np.max(spread) > max_spread:
        raise ValueError(f"Source angles unstable across training scans: {np.max(spread):.4f} deg")
    return {"method": "median elevation per ring per scan, then median across training scans",
            "kind": "empirical_estimate_not_factory_calibration", "angles_deg": angles.tolist(),
            "per_ring_spread_deg": spread.tolist(), "min_range_m": min_range,
            "max_allowed_spread_deg": max_spread, "scans": records,
            "train_ids_sha256": sha256_file(train_path)}


def unique_match(source, target, max_error=2.):
    """Minimum-total-absolute-error sorted injective assignment (dynamic programming).
    Sorted optimal matching exists for 1D absolute costs. Enforce the error
    bound during assignment, rather than rejecting a feasible solution later.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 1 or target.ndim != 1 or not len(target)
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.any(np.diff(target) <= 0) or not np.isfinite(max_error) or max_error <= 0):
        raise ValueError("Invalid angles or error bound")
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    n, m = len(source), len(target)
    if m > n:
        raise ValueError("More target beams than available source rings")
    cost = np.full((m+1, n+1), np.inf)
    take = np.zeros((m+1, n+1), dtype=bool)
    cost[0, :] = 0.
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost[i, j] = cost[i, j-1]
            error = abs(target[i-1] - sorted_source[j-1])
            candidate = cost[i-1, j-1]+error if error <= max_error+1e-9 else np.inf
            if candidate < cost[i, j]:
                cost[i, j], take[i, j] = candidate, True
    if not np.isfinite(cost[m, n]):
        raise ValueError(f"Cannot assign {m} distinct source rings within {max_error:g} deg; "
                         "source angular coverage/density is insufficient")
    selected = []
    i, j = m, n
    while i:
        if take[i, j]:
            selected.append(int(order[j-1])); i -= 1
        j -= 1
    return np.asarray(selected[::-1], dtype=int)


def make_plan(profile, calibration, max_error=2.):
    source = np.asarray(calibration["angles_deg"])
    target = np.asarray(profile["angles_deg"])
    low, high = float(target[0]), float(target[-1])
    if low < source.min()-1e-4 or high > source.max()+1e-4:
        raise ValueError(f"{profile['name']} FOV [{low:g},{high:g}] exceeds source "
                         f"[{source.min():g},{source.max():g}]; cannot synthesize unsupported rays")
    eligible = np.flatnonzero((source >= low-1e-4) & (source <= high+1e-4))
    selected = eligible[unique_match(source[eligible], target, max_error)]
    errors = np.abs(source[selected]-target)
    return {"version": VERSION, "profile": profile, "source_calibration": calibration,
            "selected_source_rings": selected.tolist(), "selected_angles_deg": source[selected].tolist(),
            "fov_source_rings": eligible.tolist(), "fov_deg": [low, high],
            "mean_abs_error_deg": float(errors.mean()), "max_abs_error_deg": float(errors.max()),
            "per_beam_abs_error_deg": errors.tolist(), "max_allowed_error_deg": max_error,
            "assignment": "minimum total absolute elevation error, distinct rings, inside target FOV",
            "scope": "angular-profile approximation; original XYZ/intensity/azimuth/timing retained"}


def select_points(points, selected):
    return points[np.isin(ring_ids(points), selected), :4]


def reference_truth(reference):
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest.get("beam_count") != 128 or manifest.get("smoke_test_only"):
        raise ValueError("Reference must be a complete 128-beam baseline")
    with (reference / "custom_infos_val.pkl").open("rb") as stream:
        infos = pickle.load(stream)
    ids = (reference / "ImageSets/val.txt").read_text().split()
    if [str(info["point_cloud"]["lidar_idx"]) for info in infos] != ids:
        raise ValueError("Reference infos and validation IDs differ")
    return infos, ids


def write_dataset(raw, reference, destination, plan, dense_fov=False):
    infos, ids = reference_truth(reference)
    name = plan["profile"]["name"] + ("_dense_fov" if dense_fov else "")
    out = destination / name
    selected = plan["fov_source_rings"] if dense_fov else plan["selected_source_rings"]
    spec = {"version": VERSION, "sampling": "dense_fov_control" if dense_fov else "elevation_profile",
            "plan": plan, "selected_source_rings": selected,
            "truth_sha256": truth_fingerprint(infos), "reference": str(reference.resolve()),
            "raw_source": str(raw.resolve()), "reference_infos_sha256": sha256_file(reference/"custom_infos_val.pkl")}
    signature = digest_json(spec)
    if out.exists():
        check_dataset(out, reference, signature)
        print(f"Reusing verified dataset: {out}", flush=True)
        return out
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=name+".partial-", dir=destination))
    try:
        for folder in ("points", "labels", "ImageSets"):
            (temporary / folder).mkdir()
        scan_records = []
        for index, frame in enumerate(ids):
            path = raw / "lidar" / (frame + ".bin")
            points = read_raw(path)
            retained = select_points(points, selected)
            np.save(temporary / "points" / (frame + ".npy"), retained)
            shutil.copyfile(reference/"labels"/(frame+".txt"), temporary/"labels"/(frame+".txt"))
            scan_records.append({"frame": frame, "raw_points": len(points), "retained_points": len(retained),
                                 "raw_sha256": sha256_file(path),
                                 "points_sha256": sha256_file(temporary/"points"/(frame+".npy"))})
            if index % 100 == 0:
                print(f"Preparing {name}: {index+1}/{len(ids)} validation frames", flush=True)
        shutil.copyfile(reference/"custom_infos_val.pkl", temporary/"custom_infos_val.pkl")
        shutil.copyfile(reference/"ImageSets/val.txt", temporary/"ImageSets/val.txt")
        manifest = {**spec, "preparation_sha256": signature, "beam_count": len(selected),
                    "val_frames": len(ids), "evaluation_only": True, "smoke_test_only": False,
                    "median_points_per_scan": float(np.median([x["retained_points"] for x in scan_records])),
                    "scans": scan_records,
                    "label_policy": "All original validation GT retained, including zero-point objects"}
        (temporary/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        temporary.rename(out)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return out


def check_dataset(data, reference, expected_signature=None):
    manifest = json.loads((data/"manifest.json").read_text())
    infos, ids = reference_truth(reference)
    if (manifest.get("version") != VERSION or manifest.get("smoke_test_only")
            or not manifest.get("evaluation_only")
            or manifest.get("truth_sha256") != truth_fingerprint(infos)
            or (expected_signature and manifest.get("preparation_sha256") != expected_signature)):
        raise ValueError(f"Incompatible dataset manifest: {data}; choose a new output directory")
    spec_keys = ("version", "sampling", "plan", "selected_source_rings", "truth_sha256",
                 "reference", "raw_source", "reference_infos_sha256")
    if (digest_json({key: manifest[key] for key in spec_keys}) != manifest["preparation_sha256"]
            or manifest["reference_infos_sha256"] != sha256_file(reference/"custom_infos_val.pkl")):
        raise ValueError("Sampling plan or reference fingerprint changed")
    selected = manifest["selected_source_rings"]
    if (len(selected) != len(set(selected)) or manifest["beam_count"] != len(selected)
            or any(not 0 <= ring < 128 for ring in selected)):
        raise ValueError("Selected ring count/uniqueness mismatch")
    if ((data/"ImageSets/val.txt").read_text().split() != ids
            or sha256_file(data/"custom_infos_val.pkl") != sha256_file(reference/"custom_infos_val.pkl")):
        raise ValueError("Frame order or ground truth changed")
    if [entry["frame"] for entry in manifest["scans"]] != ids:
        raise ValueError("Prepared scan records do not match validation IDs")
    for entry in manifest["scans"]:
        path = data/"points"/(entry["frame"]+".npy")
        if sha256_file(path) != entry["points_sha256"]:
            raise ValueError(f"Changed prepared scan: {path}")
        raw_path = Path(manifest["raw_source"])/"lidar"/(entry["frame"]+".bin")
        if sha256_file(raw_path) != entry["raw_sha256"]:
            raise ValueError(f"Raw validation scan changed: {raw_path}")
    return manifest


COMPARISON_KEYS = ("checkpoint_sha256", "model_config_sha256", "evaluation_seed",
                   "precision", "batch_size", "score_cutoff", "nms_pre_maxsize",
                   "nms_post_maxsize", "tf32_matmul", "tf32_cudnn")


def compatible_baseline(folder, truth_hash, inference):
    if not (folder/"range_ap.json").exists():
        return None
    if not (folder/"inference_manifest.json").exists():
        raise ValueError(f"Baseline missing inference provenance: {folder}")
    saved = json.loads((folder/"inference_manifest.json").read_text())
    report = json.loads((folder/"range_ap.json").read_text())
    if (any(saved.get(k) != inference.get(k) for k in COMPARISON_KEYS)
            or report.get("truth_sha256") != truth_hash
            or saved.get("prediction_sha256") != sha256_file(folder/"result.pkl")
            or report.get("prediction_sha256") != saved["prediction_sha256"]):
        raise ValueError(f"Incompatible modulo baseline: {folder}")
    return report


def summarize(base, epoch):
    """Separate CSV for angular conditions, never legacy beam-only Excel rows."""
    root = base/"eval"/f"epoch_{epoch}"/"val"
    rows = []
    shared_inference = None
    for folder in sorted(root.glob("elevation_*")):
        if not folder.is_dir() or not (folder/"range_ap.json").exists():
            continue
        manifest = json.loads((folder/"sampling_manifest.json").read_text())
        inference = json.loads((folder/"inference_manifest.json").read_text())
        report = json.loads((folder/"range_ap.json").read_text())
        if (inference["prediction_sha256"] != sha256_file(folder/"result.pkl")
                or report.get("prediction_sha256") != inference["prediction_sha256"]
                or report.get("truth_sha256") != manifest["truth_sha256"]):
            raise ValueError(f"Stale or mismatched results: {folder}")
        actual_manifest = Path(inference["data_path"])/"manifest.json"
        if sha256_file(actual_manifest) != sha256_file(folder/"sampling_manifest.json"):
            raise ValueError(f"Sampling provenance differs from evaluated dataset: {folder}")
        key = tuple(inference.get(k) for k in COMPARISON_KEYS)
        if shared_inference is not None and key != shared_inference:
            raise ValueError("Angular conditions used different checkpoints/inference settings")
        shared_inference = key
        plan = manifest["plan"]
        dense = compatible_baseline(root/"beam128", manifest["truth_sha256"], inference)
        modulo = compatible_baseline(root/f"beam{manifest['beam_count']}",
                                     manifest["truth_sha256"], inference)
        for label, metrics in report["ranges"].items():
            row = {"condition": folder.name, "profile": plan["profile"]["name"],
                   "sampling": manifest["sampling"], "beam_count": manifest["beam_count"],
                   "range": label, "mAP40": metrics["mAP40"],
                   "mean_angle_error_deg": plan["mean_abs_error_deg"] if manifest["sampling"] == "elevation_profile" else "",
                   "max_angle_error_deg": plan["max_abs_error_deg"] if manifest["sampling"] == "elevation_profile" else "",
                   "median_points_per_scan": manifest["median_points_per_scan"],
                   "checkpoint_sha256": inference["checkpoint_sha256"],
                   "sampling_sha256": manifest["preparation_sha256"]}
            dense_ap = dense["ranges"][label]["mAP40"] if dense else None
            modulo_ap = modulo["ranges"][label]["mAP40"] if modulo else None
            row["dense128_mAP40"] = dense_ap
            row["modulo_same_beam_mAP40"] = modulo_ap
            row["delta_vs_dense128_pp"] = metrics["mAP40"]-dense_ap if dense_ap is not None and metrics["mAP40"] is not None else None
            row["delta_vs_modulo_same_beam_pp"] = metrics["mAP40"]-modulo_ap if modulo_ap is not None and metrics["mAP40"] is not None else None
            for name, values in metrics["classes"].items():
                row[name+"_AP40"] = values["AP40"]
            rows.append(row)
    if not rows:
        raise ValueError("No completed elevation results found")
    output = base/"elevation"/f"epoch_{epoch}"
    output.mkdir(parents=True, exist_ok=True)
    with (output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Angular results: {output/'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "plan", "prepare", "check", "summarize"])
    parser.add_argument("--profiles", nargs="+", default=["vlp16", "vlp32c"])
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--raw", type=Path, default=Path("/home/ducanh/datasets/boreas/boreas-objects-v1"))
    parser.add_argument("--reference", type=Path, default=Path("/home/ducanh/datasets/boreas_openpcdet/128"))
    parser.add_argument("--destination", type=Path, default=Path("/home/ducanh/datasets/boreas_openpcdet/elevation"))
    parser.add_argument("--calibration-frames", type=int, default=20)
    parser.add_argument("--min-range", type=float, default=5.)
    parser.add_argument("--max-angle-error", type=float, default=2.)
    parser.add_argument("--with-fov-controls", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--epoch", type=int, default=80)
    args = parser.parse_args()
    if args.command == "list":
        for name in json.loads(args.catalog.read_text())["profiles"]:
            p = load_profile(name, args.catalog)
            print(f"{name}: {len(p['angles_deg'])} beams, [{p['angles_deg'][0]:.3f}, {p['angles_deg'][-1]:.3f}] deg; {p['description']}")
        return
    if args.command == "summarize":
        if args.base is None:
            parser.error("summarize requires --base")
        summarize(args.base, args.epoch); return
    if args.command == "check":
        if args.data is None:
            parser.error("check requires --data")
        print(check_dataset(args.data, args.reference)["beam_count"]); return
    reference_truth(args.reference)
    profiles = [load_profile(name, args.catalog) for name in args.profiles]
    calibration = estimate_source(args.raw, args.reference, args.calibration_frames, args.min_range)
    # Validate every profile before writing validation datasets.
    plans = [make_plan(p, calibration, args.max_angle_error) for p in profiles]
    for plan in plans:
        print(json.dumps({k: plan[k] for k in ["profile", "selected_source_rings", "selected_angles_deg",
                         "fov_deg", "mean_abs_error_deg", "max_abs_error_deg"]}, indent=2))
        if args.command == "prepare":
            write_dataset(args.raw, args.reference, args.destination, plan)
            if args.with_fov_controls:
                write_dataset(args.raw, args.reference, args.destination, plan, dense_fov=True)


if __name__ == "__main__":
    main()
