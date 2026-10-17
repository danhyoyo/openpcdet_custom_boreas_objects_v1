"""Compute class and range AP40 from OpenPCDet result.pkl and Boreas val infos.

This is a local, center-range-stratified 3D IoU metric, separate from the
official Boreas hidden-test score. No CUDA extension is required.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
RANGES = (("all", 0, float("inf")), ("0-20m", 0, 20),
          ("20-40m", 20, 40), ("40-60m", 40, 60),
          ("60m+", 60, float("inf")))


def corners(box: np.ndarray) -> np.ndarray:
    dx, dy, yaw = float(box[3]), float(box[4]), float(box[6])
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]]) * 0.5
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + box[:2]


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) -
                            polygon[:, 1] * np.roll(polygon[:, 0], -1))) * 0.5)


def polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject.tolist()
    for index in range(len(clip)):
        start, end = clip[index], clip[(index + 1) % len(clip)]

        def signed(point):
            edge = end - start
            offset = np.asarray(point) - start
            return float(edge[0] * offset[1] - edge[1] * offset[0])

        source, output = output, []
        if not source:
            break
        previous = source[-1]
        for current in source:
            before, after = signed(previous), signed(current)
            if (before >= 0) != (after >= 0):
                fraction = before / (before - after)
                output.append((np.asarray(previous) + fraction *
                               (np.asarray(current) - previous)).tolist())
            if after >= 0:
                output.append(current)
            previous = current
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def iou3d(first: np.ndarray, second: np.ndarray) -> float:
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return 0.0
    if np.any(first[3:6] <= 0) or np.any(second[3:6] <= 0):
        return 0.0
    z_overlap = max(0.0, min(first[2] + first[5] / 2, second[2] + second[5] / 2) -
                    max(first[2] - first[5] / 2, second[2] - second[5] / 2))
    if z_overlap == 0:
        return 0.0
    radius = (np.hypot(first[3], first[4]) + np.hypot(second[3], second[4])) / 2
    if np.linalg.norm(first[:2] - second[:2]) > radius:
        return 0.0
    overlap = polygon_area(polygon_intersection(corners(first), corners(second))) * z_overlap
    union = float(np.prod(first[3:6]) + np.prod(second[3:6]) - overlap)
    return overlap / union if union > 0 else 0.0


def in_range(box: np.ndarray, low: float, high: float) -> bool:
    distance = float(np.hypot(box[0], box[1]))
    return low <= distance < high


def ap40_from_ranked(true_positives, false_positives, positives: int) -> float | None:
    """Return interpolated AP40 (%) for an already score-ranked detection list."""
    tp = np.asarray(true_positives, dtype=np.int64)
    fp = np.asarray(false_positives, dtype=np.int64)
    if positives < 0 or tp.shape != fp.shape or tp.ndim != 1:
        raise ValueError("Invalid AP inputs")
    if np.any((tp < 0) | (tp > 1)) or np.any((fp < 0) | (fp > 1)) or np.any(tp + fp != 1):
        raise ValueError("Each ranked detection must be exactly one TP or FP")
    if positives == 0:
        return None
    if not len(tp):
        return 0.0
    recall = np.cumsum(tp) / positives
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = []
    for level in np.arange(1, 41) / 40:
        index = int(np.searchsorted(recall, level, side="left"))
        sampled.append(float(envelope[index]) if index < len(envelope) else 0.0)
    return round(float(np.mean(sampled)) * 100, 4)


def ap_for_class(predictions: dict, truth: dict, class_name: str,
                 threshold: float, low: float, high: float) -> dict:
    gt = {}
    misc = {}
    positives = 0
    detections = []
    for frame_id, record in truth.items():
        names = record["name"]
        boxes = record["gt_boxes_lidar"]
        gt[frame_id] = [box for name, box in zip(names, boxes)
                        if name == class_name and in_range(box, low, high)]
        misc[frame_id] = [box for name, box in zip(names, boxes) if name == "Misc"]
        positives += len(gt[frame_id])
    for frame_id, record in predictions.items():
        for index, name in enumerate(record["name"]):
            box = record["boxes_lidar"][index]
            if name == class_name and in_range(box, low, high):
                detections.append((float(record["score"][index]), frame_id, index, box))
    detections.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched = {frame_id: set() for frame_id in truth}
    true_positives, false_positives = [], []
    ignored = 0
    selected_tp = selected_fp = 0
    for score, frame_id, _, box in detections:
        candidates = gt[frame_id]
        overlaps = [iou3d(box, candidate) if index not in matched[frame_id] else -1
                    for index, candidate in enumerate(candidates)]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] >= threshold:
            matched[frame_id].add(best)
            selected_tp += int(score >= 0.1)
            true_positives.append(1)
            false_positives.append(0)
        elif any(iou3d(box, other) >= threshold for other in misc[frame_id]):
            ignored += 1
        else:
            selected_fp += int(score >= 0.1)
            true_positives.append(0)
            false_positives.append(1)
    precision_at_score = selected_tp / max(selected_tp + selected_fp, 1)
    recall_at_score = selected_tp / positives if positives else None
    operating = {"score_threshold": 0.1, "TP": selected_tp, "FP": selected_fp,
                 "FN": positives - selected_tp, "precision": precision_at_score,
                 "recall": recall_at_score,
                 "F1": 2 * precision_at_score * recall_at_score / max(precision_at_score + recall_at_score, 1e-12)
                       if recall_at_score is not None else None}
    return {"AP40": ap40_from_ranked(true_positives, false_positives, positives), "gt": positives,
            "predictions": len(detections), "ignored_misc": ignored, "operating_point": operating}


def evaluate_records(predictions_list: list, infos: list,
                     prediction_label: str = "in-memory") -> dict:
    truth = {str(info["point_cloud"]["lidar_idx"]): info["annos"] for info in infos}
    predictions = {str(item["frame_id"]): item for item in predictions_list}
    if set(predictions) != set(truth) or len(predictions_list) != len(predictions):
        raise ValueError("Prediction frame IDs must match validation infos exactly")
    for frame_id, record in predictions.items():
        names = np.asarray(record["name"])
        scores = np.asarray(record["score"])
        boxes = np.asarray(record["boxes_lidar"])
        if scores.shape != (len(names),) or boxes.shape != (len(names), 7):
            raise ValueError(f"Malformed prediction arrays: {frame_id}")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid prediction scores: {frame_id}")
        if not np.isfinite(boxes).all() or (len(boxes) and np.any(boxes[:, 3:6] <= 0)):
            raise ValueError(f"Invalid predicted boxes: {frame_id}")
    result = {"metric": "3D AP40 by radial center distance (local validation)",
              "prediction_file": prediction_label, "iou_thresholds": CLASSES,
              "protocol_version": "local_ap40_v1",
              "matching_policy": "Score-descending greedy one-to-one same-class 3D IoU matching; AP sampled at recall 1/40 through 1",
              "ignore_policy": "Unmatched detections overlapping any Misc box at the class IoU threshold are ignored for all target classes",
              "unobserved_gt_policy": "Target ground truth remains evaluable even when the retained input contains zero points inside its box",
              "range_policy": "GT and prediction centers independently filtered; lower inclusive, upper exclusive",
              "ranges": {}}
    for label, low, high in RANGES:
        print(f"Evaluating local AP40: {label}", flush=True)
        per_class = {name: ap_for_class(predictions, truth, name, threshold, low, high)
                     for name, threshold in CLASSES.items()}
        valid = [entry["AP40"] for entry in per_class.values() if entry["AP40"] is not None]
        result["ranges"][label] = {"classes": per_class,
                                   "mAP40": round(float(np.mean(valid)), 4) if valid else None}
    return result


def evaluate(prediction_path: Path, info_path: Path) -> dict:
    with info_path.open("rb") as stream:
        infos = pickle.load(stream)
    with prediction_path.open("rb") as stream:
        predictions_list = pickle.load(stream)
    result = evaluate_records(predictions_list, infos, str(prediction_path))
    result["info_file"] = str(info_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--infos", type=Path, default=Path(
        "/home/ducanh/datasets/boreas_openpcdet/128/custom_infos_val.pkl"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.infos)
    content = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
