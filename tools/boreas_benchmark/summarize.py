"""Collect detector results and draw vector Pareto figures without extra packages."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

MODELS = ("pointpillar", "second", "centerpoint", "voxelnext",
          "pv_rcnn", "pointpillar_lite")


def gather(root: Path, tag: str) -> list[dict]:
    rows = []
    comparison_signature = None
    for model in MODELS:
        folder = root / model / tag
        accuracy = folder / "range_ap.json"
        profile = folder / "profile.json"
        if not accuracy.is_file() or not profile.is_file():
            continue
        ap = json.loads(accuracy.read_text())
        timing = json.loads(profile.read_text())
        signature = (ap.get("protocol_version"), timing["precision"], timing["gpu"],
                     timing.get("tf32_matmul"), timing.get("tf32_cudnn"))
        if ap.get("protocol_version") != "local_ap40_v1":
            raise ValueError(f"Re-evaluate stale metric report for {model}/{tag}")
        if comparison_signature is None:
            comparison_signature = signature
        elif signature != comparison_signature:
            raise ValueError("Comparison requires the same AP protocol, inference precision, GPU and TF32 settings")
        row = {
            "model": model,
            "mAP40": ap["ranges"]["all"]["mAP40"],
            "FPS": timing["fps_from_mean_end_to_end"],
            "latency_mean_ms": timing["end_to_end"]["mean_ms"],
            "active_voxels_median": timing["active_voxels_median"],
            "precision": timing["precision"],
            "latency_p50_ms": timing["end_to_end"]["p50_ms"],
            "latency_p95_ms": timing["end_to_end"]["p95_ms"],
            "GPU_MiB": timing["peak_cuda_allocated_mib"],
            "parameters": timing["parameters"],
            "forward_including_nms_mean_ms": timing["model_forward_including_postprocess"]["mean_ms"],
        }
        for class_name, values in ap["ranges"]["all"]["classes"].items():
            row[class_name + "_AP40"] = values["AP40"]
        for range_name in ("0-20m", "20-40m", "40-60m", "60m+"):
            row["AP_" + range_name] = ap["ranges"][range_name]["mAP40"]
            for name, values in ap["ranges"][range_name]["classes"].items():
                row["AP_" + range_name + "_" + name] = values["AP40"]
        for beam in (64, 32, 16):
            path = folder / f"range_ap_{beam}.json"
            row[f"mAP40_{beam}beam"] = (json.loads(path.read_text())["ranges"]["all"]["mAP40"]
                                            if path.is_file() else None)
            reduced = row[f"mAP40_{beam}beam"]
            row[f"drop_128_to_{beam}_pp"] = (round(row["mAP40"] - reduced, 4)
                                              if reduced is not None else None)
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No model has both range_ap.json and profile.json")
    return rows


def frontier(rows: list[dict], x_key: str, maximize_x: bool) -> set[str]:
    selected = set()
    for row in rows:
        dominated = any(
            other is not row
            and (other[x_key] >= row[x_key] if maximize_x else other[x_key] <= row[x_key])
            and other["mAP40"] >= row["mAP40"]
            and (other[x_key] != row[x_key] or other["mAP40"] != row["mAP40"])
            for other in rows
        )
        if not dominated:
            selected.add(row["model"])
    return selected


def svg_plot(rows: list[dict], x_key: str, x_label: str,
             maximize_x: bool, destination: Path) -> None:
    width, height = 900, 620
    left, top, right, bottom = 105, 55, 85, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin = min(float(row[x_key]) for row in rows)
    xmax = max(float(row[x_key]) for row in rows)
    ymin = min(float(row["mAP40"]) for row in rows)
    ymax = max(float(row["mAP40"]) for row in rows)
    xpad = max((xmax - xmin) * 0.09, 1.0)
    ypad = max((ymax - ymin) * 0.12, 2.0)
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, max(0, ymin - ypad), min(100, ymax + ypad)

    def px(value):
        return left + (float(value) - xmin) / (xmax - xmin) * plot_w

    def py(value):
        return top + (ymax - float(value)) / (ymax - ymin) * plot_h

    chosen = frontier(rows, x_key, maximize_x)
    frontier_rows = sorted((row for row in rows if row["model"] in chosen),
                           key=lambda row: row[x_key])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial,sans-serif" fill="#233244">',
    ]
    for step in range(6):
        xval = xmin + (xmax - xmin) * step / 5
        yval = ymin + (ymax - ymin) * step / 5
        x, y = px(xval), py(yval)
        lines += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e6ebef"/>',
                  f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6ebef"/>',
                  f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14">{xval:.1f}</text>',
                  f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-size="14">{yval:.1f}</text>']
    lines += [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#324454" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#324454" stroke-width="2"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 28}" text-anchor="middle" font-size="18">{escape(x_label)}</text>',
        f'<text transform="translate(30 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="18">mAP40 (%)</text>',
    ]
    if len(frontier_rows) > 1:
        points = " ".join(f'{px(row[x_key]):.1f},{py(row["mAP40"]):.1f}' for row in frontier_rows)
        lines.append(f'<polyline points="{points}" fill="none" stroke="#d35425" stroke-width="2.5" stroke-dasharray="8 5"/>')
    for row in rows:
        x, y = px(row[x_key]), py(row["mAP40"])
        color = "#d35425" if row["model"] in chosen else "#267fa1"
        lines += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>',
                  f'<text x="{x + 11:.1f}" y="{y - 10:.1f}" font-size="14">{escape(row["model"])}</text>']
    lines += ['</g>', '</svg>']
    destination.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("../output/boreas_models"))
    parser.add_argument("--tag", default="seed42_beam128")
    parser.add_argument("--out", type=Path, default=Path("../output/boreas_summary"))
    args = parser.parse_args()
    rows = gather(args.root, args.tag)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tables = {"accuracy_efficiency.csv": [key for key in rows[0] if not key.startswith(("AP_", "mAP40_", "drop_"))],
              "accuracy_by_range.csv": ["model"] + [key for key in rows[0] if key.startswith("AP_")],
              "robustness.csv": ["model", "mAP40"] + [key for key in rows[0] if key.startswith(("mAP40_", "drop_"))]}
    for filename, keys in tables.items():
        with (args.out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    svg_plot(rows, "FPS", "End-to-end FPS (batch 1)", True, args.out / "pareto_fps_map.svg")
    svg_plot(rows, "GPU_MiB", "Peak allocated CUDA memory (MiB)", False,
             args.out / "pareto_memory_map.svg")
    print(f"Wrote {len(rows)} models to {args.out}")


if __name__ == "__main__":
    main()
