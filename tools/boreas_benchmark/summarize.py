"""Collect detector results and draw vector Pareto figures without extra packages."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

MODELS = ("pointpillar", "second", "centerpoint", "voxelnext",
          "pv_rcnn", "pointpillar_lite", "dsvt", "pr_dsvt")


def gather(root: Path, tag: str, run_tags=None) -> list[dict]:
    rows = []
    comparison_signature = None
    for model in MODELS:
        if run_tags is not None and model not in run_tags:
            continue
        actual_tag = run_tags[model] if run_tags is not None else tag
        folder = root / model / actual_tag
        accuracy = folder / "range_ap.json"
        profile = folder / "profile_128.json"
        if not profile.is_file():
            profile = folder / "profile.json"
        if not accuracy.is_file() or not profile.is_file():
            continue
        ap = json.loads(accuracy.read_text())
        timing = json.loads(profile.read_text())
        signature = (ap.get("protocol_version"), timing["precision"], timing["gpu"],
                     timing.get("tf32_matmul"), timing.get("tf32_cudnn"), timing.get("torch_version"),
                     timing.get("cuda_version"), timing.get("frames"), timing.get("warmup_frames"),
                     timing.get("repeats", 1))
        if ap.get("protocol_version") != "local_ap40_v1":
            raise ValueError(f"Re-evaluate stale metric report for {model}/{tag}")
        if comparison_signature is None:
            comparison_signature = signature
        elif signature != comparison_signature:
            raise ValueError("Comparison requires the same AP protocol, precision, GPU, TF32, software and profiling sample settings")
        row = {
            "model": model, "tag": actual_tag,
            "mAP40": ap["ranges"]["all"]["mAP40"],
            "FPS": timing["fps_from_mean_end_to_end"],
            "latency_mean_ms": timing["end_to_end"]["mean_ms"],
            "active_voxels_median": timing["active_voxels_median"],
            "precision": timing["precision"],
            "latency_p50_ms": timing["end_to_end"]["p50_ms"],
            "latency_p95_ms": timing["end_to_end"]["p95_ms"],
            "GPU_MiB": timing["peak_cuda_allocated_mib"],
            "GPU_reserved_MiB": timing.get("peak_cuda_reserved_mib"),
            "input_points_median": timing.get("input_points", {}).get("p50"),
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
    parser.add_argument('--runs', type=Path, help='Optional CSV model,tag for one run/model with different folder names; descriptive, not seed audited')
    args = parser.parse_args()
    run_tags = None
    if args.runs:
        with args.runs.open() as stream:
            records = list(csv.DictReader(stream))
        if len({r['model'] for r in records}) != len(records):
            raise ValueError('One-run summary requires exactly one tag/model; use study.py for multiple seeds')
        if any(r['model'] not in MODELS for r in records):
            raise ValueError('Unknown model in run CSV')
        run_tags = {r['model']: r['tag'] for r in records}
    rows = gather(args.root, args.tag, run_tags)
    try:
        from .study import ranking_analysis, write_csv
    except ImportError:
        from study import ranking_analysis, write_csv
    scores = {'beam128': {r['model']: r['mAP40'] for r in rows}}
    density, calibration = [], []
    extended_signature = None
    for row in rows:
        folder = args.root / row['model'] / row['tag']
        for beam in (128, 64, 32, 16):
            ap_path = folder / ('range_ap.json' if beam == 128 else f'range_ap_{beam}.json')
            if ap_path.exists():
                ap = json.loads(ap_path.read_text())
                if ap.get('protocol_version') != 'local_ap40_v1':
                    raise ValueError(f'Stale beam metric: {ap_path}')
                entry = ap['ranges']['all']
                scores.setdefault(f'beam{beam}', {})[row['model']] = entry['mAP40']
                for name, cls in entry['classes'].items():
                    if cls['AP40'] is not None:
                        scores.setdefault(f'beam{beam}_{name}', {})[row['model']] = cls['AP40']
                if beam == 128:
                    for band, range_entry in ap['ranges'].items():
                        if band == 'all': continue
                        scores.setdefault(f'range_{band}', {})[row['model']] = range_entry['mAP40']
                        for name, cls in range_entry['classes'].items():
                            if cls['AP40'] is not None:
                                scores.setdefault(f'range_{band}_{name}', {})[row['model']] = cls['AP40']
            path = folder / f'extended_{beam}.json'
            if not path.exists(): continue
            extra = json.loads(path.read_text())
            if extra.get('protocol_version') != 'density_calibration_v1':
                raise ValueError(f'Stale extended metric: {path}')
            signature = (extra['truth_sha256'], extra['calibration_bins'], extra['operating_score'],
                         json.dumps(extra['density_bins']), extra['verified_inference'])
            if extended_signature is None:
                extended_signature = signature
            elif signature != extended_signature:
                raise ValueError('Extended reports differ in GT, binning, score threshold or verification status')
            for name, entry in extra['classes'].items():
                calibration.append({'model':row['model'],'beam':beam,'class':name,
                                    **{k:v for k,v in entry['calibration'].items() if k!='bins'},
                                    'recall':entry['operating_point']['recall'],
                                    'verified_inference':extra['verified_inference']})
                for mode in ('density_fixed128','density_actual'):
                    for band, groups in entry[mode].items():
                        for group, counts in groups.items():
                            density.append({'model':row['model'],'beam':beam,'class':name,
                                            'grouping':mode,'range':band,'points':group,**counts})
    args.out.mkdir(parents=True, exist_ok=True)
    ranking = ranking_analysis(scores)
    ranking['note'] = 'One-tag descriptive comparison; folder tags do not establish a common training seed. Use study.py for audited multi-seed analysis.'
    (args.out / 'ranking_stability.json').write_text(json.dumps(ranking, indent=2) + '\n')
    write_csv(args.out / 'density_recall.csv', density)
    write_csv(args.out / 'calibration.csv', calibration)
    sparse_efficiency = []
    profile_signatures = {}
    for row in rows:
        folder = args.root / row['model'] / row['tag']
        for beam in (128, 64, 32, 16):
            profile_path = folder / f'profile_{beam}.json'
            if beam == 128 and not profile_path.exists():
                profile_path = folder / 'profile.json'
            ap_path = folder / ('range_ap.json' if beam == 128 else f'range_ap_{beam}.json')
            if not profile_path.exists() or not ap_path.exists():
                continue
            timing = json.loads(profile_path.read_text())
            ap = json.loads(ap_path.read_text())
            signature = (timing['precision'], timing['gpu'], timing.get('tf32_matmul'),
                         timing.get('tf32_cudnn'), timing.get('torch_version'),
                         timing.get('cuda_version'), timing.get('frames'),
                         timing.get('warmup_frames'), timing.get('repeats', 1))
            if beam in profile_signatures and signature != profile_signatures[beam]:
                raise ValueError(f'Beam-{beam} profiles use incompatible measurement conditions')
            profile_signatures[beam] = signature
            sparse_efficiency.append({
                'model': row['model'], 'tag': row['tag'], 'beam': beam,
                'mAP40': ap['ranges']['all']['mAP40'],
                'latency_mean_ms': timing['end_to_end']['mean_ms'],
                'latency_p50_ms': timing['end_to_end']['p50_ms'],
                'latency_p95_ms': timing['end_to_end']['p95_ms'],
                'FPS': timing['fps_from_mean_end_to_end'],
                'GPU_allocated_MiB': timing['peak_cuda_allocated_mib'],
                'GPU_reserved_MiB': timing.get('peak_cuda_reserved_mib'),
                'input_points_mean': timing.get('input_points', {}).get('mean'),
                'input_points_p50': timing.get('input_points', {}).get('p50'),
                'active_voxels_mean': timing.get('active_voxels', {}).get('mean'),
                'active_voxels_p50': timing.get('active_voxels', {}).get(
                    'p50', timing.get('active_voxels_median')),
                'repeats': timing.get('repeats', 1), 'frames': timing.get('frames')})
    write_csv(args.out / 'sparsity_efficiency.csv', sparse_efficiency)
    for beam in (128, 64, 32, 16):
        selected = [item for item in sparse_efficiency if item['beam'] == beam]
        if len(selected) >= 2:
            svg_plot(selected, 'FPS', f'End-to-end FPS at {beam} rings (batch 1)', True,
                     args.out / f'pareto_fps_map_{beam}ring.svg')
            memory_rows = [{**item, 'GPU_MiB': item['GPU_allocated_MiB']}
                           for item in selected]
            svg_plot(memory_rows, 'GPU_MiB',
                     f'Peak allocated CUDA memory at {beam} rings (MiB)', False,
                     args.out / f'pareto_memory_map_{beam}ring.svg')
    sensitivity_rows, sensitivity_scores = [], {}
    sensitivity_grid = None
    for row in rows:
        path = args.root / row['model'] / row['tag'] / 'sensitivity_128.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report.get('protocol_version') != 'local_ap40_export_sensitivity_v1':
            raise ValueError(f'Stale export-sensitivity report: {path}')
        grid = {(entry['score_cutoff'], entry['output_cap'])
                for entry in report['conditions']}
        if sensitivity_grid is None:
            sensitivity_grid = grid
        elif grid != sensitivity_grid:
            raise ValueError('Export-sensitivity reports use different cutoff/cap grids')
        for entry in report['conditions']:
            key = f"cutoff={entry['score_cutoff']:g},cap={entry['output_cap']}"
            sensitivity_scores.setdefault(key, {})[row['model']] = entry['mAP40']
            sensitivity_rows.append({'model': row['model'], 'tag': row['tag'], **entry})
    write_csv(args.out / 'export_sensitivity.csv', sensitivity_rows)
    if sensitivity_scores:
        sensitivity_ranking = ranking_analysis(sensitivity_scores)
        sensitivity_ranking['scope'] = (
            'Stricter cutoffs/additional global post-export truncation on saved post-NMS '
            'predictions; configured NMS-cap changes and reverse directions require fresh inference.'
        )
        (args.out / 'export_sensitivity_ranking.json').write_text(
            json.dumps(sensitivity_ranking, indent=2) + '\n')
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
