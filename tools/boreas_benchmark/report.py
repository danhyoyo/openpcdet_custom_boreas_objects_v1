"""Persist accuracy metrics and optional local radial-distance evaluation."""
import json
from pathlib import Path
import numpy as np
from boreas_benchmark.evaluate import evaluate


def write_reports(dataset, result_dir, metrics, range_eval, logger):
    result_dir = Path(result_dir)
    serializable = {key: float(value) for key, value in metrics.items()
                    if isinstance(value, (int, float, np.number))}
    classes = {}
    for name in dataset.class_names:
        key = name + "_3d/moderate_R40"
        if key in serializable:
            classes[name] = serializable[key]
    accuracy = {"protocol": "KITTI evaluator 3D AP_R40; supplementary only: synthetic 2D boxes, no real difficulty tiers",
                "classes": classes, "mAP_R40": float(np.mean(list(classes.values())))
                if len(classes) == len(dataset.class_names) else None,
                "metrics": serializable}
    (result_dir / "kitti_accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    if not range_eval:
        accuracy = {"protocol": "Local AP40 not requested; rerun evaluation with --range_eval",
                    "classes": {}, "mAP40": None, "supplementary_file": "kitti_accuracy.json"}
    if range_eval:
        infos = dataset.root_path / ("custom_infos_" + dataset.split + ".pkl")
        report = evaluate(result_dir / "result.pkl", infos)
        (result_dir / "range_ap.json").write_text(json.dumps(report, indent=2) + "\n")
        accuracy = {"protocol": report["metric"], "protocol_version": report["protocol_version"],
                    "classes": {name: entry["AP40"] for name, entry in report["ranges"]["all"]["classes"].items()},
                    "mAP40": report["ranges"]["all"]["mAP40"],
                    "iou_thresholds": report["iou_thresholds"], "ignore_policy": report["ignore_policy"]}
        for label, values in report["ranges"].items():
            logger.info("Local AP40 %s: %s", label, values)
    (result_dir / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
