"""Audit local AP40 sensitivity to stricter cutoff/post-export truncation.

This CPU-only audit operates on saved post-NMS predictions. It can emulate a
higher cutoff or an additional global per-frame cap on the saved detections.
It does not emulate architecture-specific pre/post-NMS cap semantics. Lower
cutoffs and any configured NMS-cap change require fresh inference.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

try:
    from .evaluate import CLASSES, ap_for_class
    from .extended import sha256_file
except ImportError:
    from evaluate import CLASSES, ap_for_class
    from extended import sha256_file

VERSION = "local_ap40_export_sensitivity_v1"


def filter_predictions(records: list, cutoff: float, cap: int) -> list:
    if not 0 <= cutoff <= 1 or cap < 1:
        raise ValueError("Cutoff must be in [0,1] and cap must be positive")
    filtered = []
    for record in records:
        names = np.asarray(record["name"])
        boxes = np.asarray(record["boxes_lidar"])
        scores = np.asarray(record["score"], dtype=float)
        if scores.shape != (len(names),) or boxes.shape != (len(names), 7):
            raise ValueError(f"Malformed prediction arrays: {record.get('frame_id')}")
        order = sorted((i for i, score in enumerate(scores) if score >= cutoff),
                       key=lambda i: (-scores[i], i))[:cap]
        filtered.append({"frame_id": record["frame_id"],
                         "name": names[order], "boxes_lidar": boxes[order],
                         "score": scores[order]})
    return filtered


def audit(prediction_path: Path, info_path: Path, manifest_path: Path,
          cutoffs: list[float], caps: list[int]) -> dict:
    manifest = json.loads(manifest_path.read_text())
    source_cutoff = float(manifest["score_cutoff"])
    source_cap = int(manifest["nms_post_maxsize"])
    if manifest.get("prediction_sha256") != sha256_file(prediction_path):
        raise ValueError("Prediction hash does not match inference manifest")
    if any(value < source_cutoff for value in cutoffs):
        raise ValueError("Saved predictions cannot audit a cutoff below the source cutoff")
    if any(value > source_cap for value in caps):
        raise ValueError("Saved predictions cannot audit a global cap above the source cap")
    with prediction_path.open("rb") as stream:
        records = pickle.load(stream)
    with info_path.open("rb") as stream:
        infos = pickle.load(stream)
    truth = {str(info["point_cloud"]["lidar_idx"]): info["annos"] for info in infos}
    if len(records) != len(truth) or {str(item["frame_id"]) for item in records} != set(truth):
        raise ValueError("Prediction frame IDs must match validation infos exactly")
    conditions = []
    for cutoff in sorted(set(cutoffs)):
        for cap in sorted(set(caps)):
            selected = filter_predictions(records, cutoff, cap)
            predictions = {str(item["frame_id"]): item for item in selected}
            per_class = {name: ap_for_class(predictions, truth, name, threshold,
                                            0, float("inf"))
                         for name, threshold in CLASSES.items()}
            supported = [entry["AP40"] for entry in per_class.values()
                         if entry["AP40"] is not None]
            overall_map = round(float(np.mean(supported)), 4) if supported else None
            conditions.append({"score_cutoff": cutoff, "output_cap": cap,
                               "mAP40": overall_map,
                               "classes": {name: entry["AP40"]
                                           for name, entry in per_class.items()},
                               "detections": int(sum(len(item["score"]) for item in selected))})
    return {"protocol_version": VERSION, "metric_protocol": "local_ap40_v1",
            "prediction_file": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "info_file": str(info_path.resolve()),
            "source_score_cutoff": source_cutoff, "source_output_cap": source_cap,
            "cutoffs": sorted(set(cutoffs)), "caps": sorted(set(caps)),
            "scope": "Stricter cutoff/additional global truncation on saved post-NMS predictions",
            "limitation": "Architecture-specific NMS cap changes, lower cutoffs, and higher caps require fresh inference.",
            "conditions": conditions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--infos", required=True, type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Defaults to inference_manifest.json beside predictions")
    parser.add_argument("--cutoffs", type=float, nargs="*", default=None)
    parser.add_argument("--caps", type=int, nargs="*", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.predictions.with_name("inference_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    source_cutoff, source_cap = float(manifest["score_cutoff"]), int(manifest["nms_post_maxsize"])
    cutoffs = args.cutoffs or [source_cutoff, .005, .01, .05, .1]
    caps = args.caps or sorted({min(value, source_cap) for value in (50, 100, 200, source_cap)})
    result = audit(args.predictions, args.infos, manifest_path, cutoffs, caps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote export sensitivity audit: {args.out}")


if __name__ == "__main__":
    main()
