"""Frozen-checkpoint density recall and detection confidence calibration (CPU).

No detector training changes. Counts use prepared XYZ before model voxel caps.
ECE/Brier describe nonignored saved post-NMS detections, not missing objects.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickle
from html import escape
from pathlib import Path
import numpy as np
try:
    from .evaluate import CLASSES, RANGES, iou3d
except ImportError:
    from evaluate import CLASSES, RANGES, iou3d

VERSION = "density_calibration_v1"
DENSITY_BINS = (("0", 0, 1), ("1-5", 1, 6), ("6-20", 6, 21),
                ("21-50", 21, 51), ("51-100", 51, 101), ("101+", 101, None))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth_fingerprint(infos):
    digest = hashlib.sha256()
    ids = []
    for info in infos:
        frame = str(info["point_cloud"]["lidar_idx"])
        ids.append(frame)
        digest.update(json.dumps([frame, list(map(str, info["annos"]["name"]))]).encode())
        digest.update(np.asarray(info["annos"]["gt_boxes_lidar"], dtype="<f8").tobytes())
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ground-truth frame IDs")
    return digest.hexdigest()


def load_records(prediction_path, info_path):
    with Path(info_path).open("rb") as stream:
        infos = pickle.load(stream)
    with Path(prediction_path).open("rb") as stream:
        records = pickle.load(stream)
    fingerprint = truth_fingerprint(infos)
    truth = {str(i["point_cloud"]["lidar_idx"]): i["annos"] for i in infos}
    predictions = {str(i["frame_id"]): i for i in records}
    if len(predictions) != len(records) or set(predictions) != set(truth):
        raise ValueError("Prediction IDs must match ground truth exactly, without duplicates")
    for frame, record in predictions.items():
        scores = np.asarray(record["score"])
        boxes = np.asarray(record["boxes_lidar"])
        if len(scores) != len(record["name"]) or boxes.shape != (len(scores), 7):
            raise ValueError(f"Malformed prediction arrays: {frame}")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Calibration requires finite probability scores in [0,1]")
        if not np.isfinite(boxes).all() or (len(boxes) and np.any(boxes[:, 3:6] <= 0)):
            raise ValueError(f"Invalid predicted boxes: {frame}")
    return infos, truth, predictions, fingerprint


def count_points_in_boxes(points, boxes):
    """Inclusive faces of upright rotated boxes; no cap or model voxelization."""
    xyz = np.asarray(points)[:, :3]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Point coordinates must be finite Nx3")
    counts = []
    for box in np.asarray(boxes).reshape(-1, 7):
        x, y, z, dx, dy, dz, yaw = map(float, box)
        if not np.isfinite(box).all() or min(dx, dy, dz) <= 0:
            raise ValueError("Invalid ground-truth box")
        c, s = np.cos(yaw), np.sin(yaw)
        # Cheap axis-aligned bounding envelope before rotating candidate points.
        ex, ey = (abs(c)*dx + abs(s)*dy)/2, (abs(s)*dx + abs(c)*dy)/2
        mask = ((np.abs(xyz[:, 0]-x) <= ex + 1e-6) &
                (np.abs(xyz[:, 1]-y) <= ey + 1e-6) &
                (np.abs(xyz[:, 2]-z) <= dz/2 + 1e-6))
        local = np.asarray(xyz[mask], dtype=np.float64) - [x, y, z]
        lx, ly = c*local[:, 0] + s*local[:, 1], -s*local[:, 0] + c*local[:, 1]
        counts.append(int(np.count_nonzero((np.abs(lx) <= dx/2 + 1e-6) &
                                          (np.abs(ly) <= dy/2 + 1e-6))))
    return counts


def density_counts(infos, data_path, cache_dir):
    data_path, cache_dir = Path(data_path).resolve(), Path(cache_dir)
    ground_hash = truth_fingerprint(infos)
    digest = hashlib.sha256((VERSION + str(data_path) + ground_hash).encode())
    for info in infos:
        frame = str(info["point_cloud"]["lidar_idx"])
        stat = (data_path / "points" / (frame + ".npy")).stat()
        digest.update(f"{frame}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    signature = digest.hexdigest()
    cache = cache_dir / (signature + ".json")
    if cache.exists():
        saved = json.loads(cache.read_text())
        print(f"Density cache hit: {data_path}", flush=True)
        return saved["counts"]
    counts = {}
    for index, info in enumerate(infos):
        frame = str(info["point_cloud"]["lidar_idx"])
        points = np.load(data_path / "points" / (frame + ".npy"), mmap_mode="r")
        # Shared point-domain crop, before representation-specific truncation.
        mask = ((points[:, 0] >= -75.2) & (points[:, 0] <= 75.2) &
                (points[:, 1] >= -75.2) & (points[:, 1] <= 75.2) &
                (points[:, 2] >= -3) & (points[:, 2] <= 5))
        counts[frame] = count_points_in_boxes(points[mask], info["annos"]["gt_boxes_lidar"])
        if index % 25 == 0 or index + 1 == len(infos):
            print(f"Counting points {data_path.name}: {index+1}/{len(infos)} frames", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import uuid
    temporary = cache.with_name(cache.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps({"version": VERSION, "signature": signature,
                                    "truth_sha256": ground_hash, "counts": counts}) + "\n")
    temporary.replace(cache)
    return counts


def match_class(predictions, truth, name, threshold):
    """Full-domain score-greedy matching, identical target/Misc priority to AP40."""
    target = {f: [i for i, n in enumerate(g["name"]) if n == name] for f, g in truth.items()}
    used = {f: set() for f in truth}
    detections = []
    for frame, record in predictions.items():
        for index, class_name in enumerate(record["name"]):
            if class_name == name:
                detections.append((float(record["score"][index]), frame, index, record["boxes_lidar"][index]))
    detections.sort(key=lambda d: (-d[0], d[1], d[2]))
    outcomes, matched = [], {}
    ignored = 0
    for score, frame, index, box in detections:
        gt = truth[frame]
        candidates = [i for i in target[frame] if i not in used[frame]]
        overlaps = [iou3d(box, gt["gt_boxes_lidar"][i]) for i in candidates]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] >= threshold:
            gt_index = candidates[best]
            used[frame].add(gt_index)
            matched[(frame, gt_index)] = score
            outcomes.append({"score": score, "correct": 1, "frame": frame, "index": index,
                             "iou": float(overlaps[best])})
        elif any(iou3d(box, g) >= threshold for n, g in zip(gt["name"], gt["gt_boxes_lidar"]) if n == "Misc"):
            ignored += 1
        else:
            outcomes.append({"score": score, "correct": 0, "frame": frame, "index": index, "iou": 0.0})
    return outcomes, matched, ignored


def calibration_metrics(outcomes, bins=15):
    if bins < 1:
        raise ValueError("Calibration bins must be positive")
    scores = np.asarray([o["score"] for o in outcomes], dtype=float)
    labels = np.asarray([o["correct"] for o in outcomes], dtype=float)
    ids = np.minimum((scores * bins).astype(int), bins-1)
    entries = []
    ece = 0.0
    for index in range(bins):
        mask = ids == index
        count = int(mask.sum())
        confidence = float(scores[mask].mean()) if count else None
        precision = float(labels[mask].mean()) if count else None
        if count:
            ece += count / len(scores) * abs(confidence-precision)
        entries.append({"lower": index/bins, "upper": (index+1)/bins, "count": count,
                        "mean_confidence": confidence, "empirical_precision": precision})
    return {"detections": len(scores), "ECE": ece if len(scores) else None,
            "Brier": float(np.mean((scores-labels)**2)) if len(scores) else None,
            "mean_confidence": float(scores.mean()) if len(scores) else None,
            "empirical_precision": float(labels.mean()) if len(scores) else None, "bins": entries}


def density_recall(truth, matched, counts, name, score_threshold=0.1, low=0, high=float("inf")):
    groups = {label: {"gt": 0, "TP": 0} for label, _, _ in DENSITY_BINS}
    for frame, annotations in truth.items():
        if len(counts[frame]) != len(annotations["name"]):
            raise ValueError("Density count/annotation length mismatch")
        for index, class_name in enumerate(annotations["name"]):
            if class_name != name:
                continue
            box = annotations["gt_boxes_lidar"][index]
            if not low <= float(np.hypot(*box[:2])) < high:
                continue
            count = counts[frame][index]
            label = next(label for label, a, b in DENSITY_BINS if count >= a and (b is None or count < b))
            groups[label]["gt"] += 1
            groups[label]["TP"] += int(matched.get((frame, index), -1) >= score_threshold)
    for entry in groups.values():
        entry["FN"] = entry["gt"] - entry["TP"]
        entry["recall"] = entry["TP"]/entry["gt"] if entry["gt"] else None
    return groups


def analyze(truth, predictions, dense_counts, actual_counts, bins=15, score_threshold=0.1):
    if not 0 <= score_threshold <= 1:
        raise ValueError("Operating score must be in [0,1]")
    result = {}
    for name, threshold in CLASSES.items():
        print(f"Extended matching: {name}", flush=True)
        outcomes, matched, ignored = match_class(predictions, truth, name, threshold)
        gt = sum(np.count_nonzero(record["name"] == name) for record in truth.values())
        tp = sum(score >= score_threshold for score in matched.values())
        fp = sum(o["correct"] == 0 and o["score"] >= score_threshold for o in outcomes)
        result[name] = {
            "iou_threshold": threshold, "ignored_misc": ignored,
            "operating_point": {"score_threshold": score_threshold, "gt": int(gt), "TP": int(tp),
                                "FP": int(fp), "FN": int(gt-tp), "recall": tp/gt if gt else None},
            "calibration": calibration_metrics(outcomes, bins),
            "density_fixed128": {label: density_recall(truth, matched, dense_counts, name, score_threshold, low, high)
                                 for label, low, high in RANGES},
            "density_actual": {label: density_recall(truth, matched, actual_counts, name, score_threshold, low, high)
                               for label, low, high in RANGES},
        }
    return result


def reliability_svg(metrics, title, path):
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 650 520">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="325" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(title)}</text>',
             '<line x1="75" y1="440" x2="590" y2="60" stroke="#888" stroke-dasharray="5 5"/>']
    for index in range(6):
        val = index/5; x=75+515*val; y=440-380*val
        lines += [f'<line x1="{x}" y1="60" x2="{x}" y2="440" stroke="#eee"/>',
                  f'<text x="{x}" y="462" text-anchor="middle">{val:.1f}</text>',
                  f'<text x="60" y="{y}" text-anchor="end">{val:.1f}</text>']
    for entry in metrics["bins"]:
        if entry["count"]:
            x=75+515*entry["mean_confidence"]; y=440-380*entry["empirical_precision"]
            lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#267fa1"><title>n={entry["count"]}</title></circle>')
    lines += ['<text x="325" y="502" text-anchor="middle">Mean detection confidence</text>',
              '<text transform="translate(20 250) rotate(-90)" text-anchor="middle">Empirical precision (3D TP fraction)</text>', '</svg>']
    Path(path).write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference-data", type=Path, default=Path("/home/ducanh/datasets/boreas_openpcdet/128"))
    parser.add_argument("--beam", type=int, required=True, choices=range(1,129))
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--cache-dir", type=Path, default=Path("../output/boreas_density_cache"))
    parser.add_argument("--allow-unverified", action="store_true", help="Exploration only: permit legacy predictions without an inference manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for data, expected in ((args.data, args.beam), (args.reference_data, 128)):
        prepared = data / 'manifest.json'
        if prepared.exists():
            if json.loads(prepared.read_text()).get('beam_count') != expected:
                raise ValueError('Beam label disagrees with prepared dataset manifest')
        elif data.name != str(expected) and not args.allow_unverified:
            raise ValueError('Cannot verify beam label: provide prepared manifest or use exploratory --allow-unverified')
    infos, truth, predictions, truth_hash = load_records(args.predictions, args.data/"custom_infos_val.pkl")
    with (args.reference_data/"custom_infos_val.pkl").open("rb") as stream:
        reference_infos = pickle.load(stream)
    if truth_fingerprint(reference_infos) != truth_hash:
        raise ValueError("Dense and sparse ground truth / frame order must be identical")
    manifest_path = args.predictions.with_name("inference_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    verified = (manifest is not None and manifest.get("prediction_sha256") == sha256_file(args.predictions)
                and Path(manifest["data_path"]).resolve() == args.data.resolve())
    if not verified and not args.allow_unverified:
        raise ValueError("Missing/stale inference_manifest.json: rerun test or explicitly use --allow-unverified for exploration")
    counts128 = density_counts(reference_infos, args.reference_data, args.cache_dir)
    actual = counts128 if args.data.resolve() == args.reference_data.resolve() else density_counts(infos, args.data, args.cache_dir)
    report = {"protocol_version": VERSION, "beam": args.beam, "truth_sha256": truth_hash,
              "reference_data": str(args.reference_data.resolve()), "data_path": str(args.data.resolve()),
              "prediction_file": str(args.predictions.resolve()), "verified_inference": bool(verified),
              "inference": manifest, "calibration_bins": args.bins,
              "density_bins": DENSITY_BINS, "operating_score": args.score_threshold,
              "density_note": "Full-domain matching first; recall stratified by fixed128 and actual GT point counts. Inclusive box faces. Shared spatial crop before voxel caps.",
              "calibration_note": "Equal-width ECE/Brier on nonignored post-NMS detections; no score recalibration. No detections means undefined ECE/Brier; missed objects represented by recall, not ECE.",
              "classes": analyze(truth, predictions, counts128, actual, args.bins, args.score_threshold)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    for name, entry in report["classes"].items():
        reliability_svg(entry["calibration"], f"{name}, {args.beam} rings", args.out.with_name(f"reliability_{args.beam}_{name}.svg"))
    print(f"Wrote extended report: {args.out}", flush=True)

if __name__ == "__main__":
    main()
