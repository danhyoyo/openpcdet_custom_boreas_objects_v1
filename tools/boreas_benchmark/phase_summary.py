"""Summarize modulo-phase AP for one frozen checkpoint (CPU only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASSES = ("Car", "Pedestrian", "Cyclist")


def condition_name(beams: int, phase: int) -> str:
    return f"beam{beams}" if phase == 0 else f"beam{beams}_phase{phase}"


def summarize(eval_root: Path, data_root: Path, allow_partial: bool = False) -> dict:
    rows, common = [], None
    for beams in (64, 32, 16):
        expected = set(range(128 // beams))
        found = set()
        for phase in sorted(expected):
            condition = condition_name(beams, phase)
            folder = eval_root / condition
            metric_path, inference_path = folder/"range_ap.json", folder/"inference_manifest.json"
            data = data_root / (str(beams) if phase == 0 else f"{beams}_phase{phase}")
            sampling_path = data/"manifest.json"
            if not (metric_path.exists() and inference_path.exists() and sampling_path.exists()):
                continue
            metric = json.loads(metric_path.read_text())
            inference = json.loads(inference_path.read_text())
            sampling = json.loads(sampling_path.read_text())
            if metric.get("protocol_version") != "local_ap40_v1":
                raise ValueError(f"Stale AP protocol: {metric_path}")
            if (sampling.get("sampling_strategy"), sampling.get("beam_count"),
                    sampling.get("phase")) != ("ring_id_modulo", beams, phase):
                raise ValueError(f"Invalid sampling manifest: {sampling_path}")
            expected_rings = list(range(phase, 128, 128 // beams))
            if sampling.get("retained_ring_ids") != expected_rings:
                raise ValueError(f"Unexpected retained ring IDs: {sampling_path}")
            if metric.get("prediction_sha256") != inference.get("prediction_sha256"):
                raise ValueError(f"Metric/prediction hash mismatch: {folder}")
            signature = (inference.get("checkpoint_sha256"), inference.get("model_config_sha256"),
                         inference.get("score_cutoff"), inference.get("nms_pre_maxsize"),
                         inference.get("nms_post_maxsize"), inference.get("precision"),
                         inference.get("tf32_matmul"), inference.get("tf32_cudnn"),
                         metric.get("truth_sha256"))
            if common is None:
                common = signature
            elif signature != common:
                raise ValueError("Phase reports do not use the same frozen checkpoint/inference protocol")
            overall = metric["ranges"]["all"]
            rows.append({"beams": beams, "phase": phase, "condition": condition,
                         "mAP40": overall["mAP40"],
                         **{name + "_AP40": overall["classes"][name]["AP40"]
                            for name in CLASSES}})
            found.add(phase)
        if found != expected and not allow_partial:
            raise FileNotFoundError(f"Beam {beams}: missing phases {sorted(expected-found)}")
    if not rows:
        raise FileNotFoundError("No verified modulo-phase reports found")
    aggregates = []
    for beams in (64, 32, 16):
        subset = [row for row in rows if row["beams"] == beams]
        if not subset:
            continue
        for metric in ("mAP40",) + tuple(name + "_AP40" for name in CLASSES):
            values = np.asarray([row[metric] for row in subset], dtype=float)
            aggregates.append({"beams": beams, "metric": metric, "phases": len(values),
                               "mean": float(values.mean()),
                               "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                               "min": float(values.min()), "max": float(values.max()),
                               "worst_phase": subset[int(values.argmin())]["phase"],
                               "best_phase": subset[int(values.argmax())]["phase"]})
    return {"protocol_version": "modulo_phase_summary_v1",
            "complete": all(sum(row["beams"] == b for row in rows) == 128//b
                            for b in (64, 32, 16)),
            "rows": rows, "aggregates": aggregates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/ducanh/datasets/boreas_openpcdet"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.eval_root, args.data_root, args.allow_partial)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"phase_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for filename, rows in (("phase_scores.csv", report["rows"]),
                           ("phase_aggregates.csv", report["aggregates"])):
        with (args.out/filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(f"Wrote modulo-phase summary: {args.out}")


if __name__ == "__main__":
    main()
