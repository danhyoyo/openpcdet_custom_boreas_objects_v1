"""Export compact same-frame paper figures for LiDAR sampling conditions.

The default output uses a close oblique view, common ground-truth boxes, no box
text, and one short condition title below each panel. No inference or GPU is
required.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_rings import BACKGROUND, font, render_panel

MOD = [
    ("128", "beam128", "128 beams"),
    ("64", "beam64", "64 beams"),
    ("32", "beam32", "32 beams"),
    ("16", "beam16", "16 beams"),
]
ELEV = [
    ("16", "beam16", "Modulo 16"),
    ("elevation/vlp16", "elevation_vlp16", "VLP-16 matched"),
    ("elevation/vlp16_dense_fov", "elevation_vlp16_dense_fov", "VLP-16 FOV (125 rings)"),
    ("32", "beam32", "Modulo 32"),
    ("elevation/vlp32c", "elevation_vlp32c", "VLP-32C matched"),
]
PHASE = [
    (
        "16" if phase == 0 else f"16_phase{phase}",
        "beam16" if phase == 0 else f"beam16_phase{phase}",
        f"16 beams - phase {phase}",
    )
    for phase in range(8)
]


def frame_id(root, split, fid, index):
    ids = (root / "128" / "ImageSets" / f"{split}.txt").read_text().split()
    if fid is None:
        if not 0 <= index < len(ids):
            raise ValueError("frame index outside split")
        fid = ids[index]
    if fid not in ids:
        raise ValueError(f"{fid} not in {split}")
    return fid


def gt(root, split, fid, bounds):
    data = pickle.load((root / "128" / f"custom_infos_{split}.pkl").open("rb"))
    item = next(x for x in data if str(x["point_cloud"]["lidar_idx"]) == fid)
    annos = item["annos"]
    boxes = np.asarray(annos["gt_boxes_lidar"], float)
    names = np.asarray(annos["name"])
    keep = np.isin(names, ["Car", "Pedestrian", "Cyclist"])
    keep &= ((boxes[:, :3] >= bounds[:3]) & (boxes[:, :3] <= bounds[3:])).all(1)
    return boxes[keep, :7], names[keep], np.ones(keep.sum())


def prediction(eval_root, tag, fid, threshold, bounds):
    rows = pickle.load((eval_root / tag / "result.pkl").open("rb"))
    row = next(x for x in rows if str(x["frame_id"]) == fid)
    boxes = np.asarray(row["boxes_lidar"], float)
    names = np.asarray(row["name"])
    scores = np.asarray(row["score"], float)
    keep = scores >= threshold
    keep &= ((boxes[:, :3] >= bounds[:3]) & (boxes[:, :3] <= bounds[3:])).all(1)
    return boxes[keep, :7], names[keep], scores[keep]


def scene(args, spec, fid, bounds, gt_data):
    data_rel, eval_tag, label = spec
    data = args.data_root / data_rel
    ids = (data / "ImageSets" / f"{args.split}.txt").read_text().split()
    if fid not in ids:
        raise ValueError(f"{fid} missing from {data}")
    points = np.load(data / "points" / f"{fid}.npy")
    xyz = np.asarray(points[:, :3], float)
    keep = np.isfinite(xyz).all(1)
    keep &= ((xyz >= bounds[:3]) & (xyz <= bounds[3:])).all(1)
    box_data = (
        gt_data
        if args.boxes == "gt"
        else prediction(args.eval_root, eval_tag, fid, args.score_thresh, bounds)
    )
    boxes, names, scores = box_data
    return {
        "beam": label,
        "points": xyz[keep],
        "input_points": len(points),
        "boxes": boxes,
        "names": names,
        "scores": scores,
        "bounds": bounds,
        "condition": data_rel,
        "visible_points": int(keep.sum()),
        "show_scores": args.boxes == "prediction",
        "box_kind": "detections" if args.boxes == "prediction" else "GT boxes",
    }


def captioned_panel(image, title, border=6, caption_height=92):
    width, height = image.size
    panel = Image.new("RGB", (width + 2 * border, height + 2 * border + caption_height), "white")
    panel.paste(image, (border, border))
    draw = ImageDraw.Draw(panel)
    caption_font = font(38)
    box = draw.textbbox((0, 0), title, font=caption_font)
    text_width = box[2] - box[0]
    text_y = border + height + (caption_height - (box[3] - box[1])) // 2 - box[1]
    draw.text(((panel.width - text_width) / 2, text_y), title, fill="black", font=caption_font)
    return panel


def grid(path, pdf_path, panels, cols, gap=18):
    width, height = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    output = Image.new(
        "RGB",
        (cols * width + (cols - 1) * gap, rows * height + (rows - 1) * gap),
        "white",
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, cols)
        row_count = min(cols, len(panels) - row * cols)
        row_offset = (cols - row_count) * (width + gap) // 2
        output.paste(panel, (row_offset + column * (width + gap), row * (height + gap)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, dpi=(300, 300))
    output.save(pdf_path, "PDF", resolution=300.0)
    return [path, pdf_path]


def render(args, scenes, fid, views, cols, name):
    panels = []
    for view in views:
        for item in scenes:
            image = render_panel(
                item,
                fid,
                view,
                args.score_thresh,
                size=(1000, 620),
                elevation=args.elevation,
                azimuth=args.azimuth,
                show_labels=args.box_labels,
                paper_style=True,
            )
            panels.append(captioned_panel(image, item["beam"]))
    png_path = args.out / name
    pdf_path = args.pdf_out / Path(name).with_suffix(".pdf")
    return grid(png_path, pdf_path, panels, cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/ducanh/datasets/boreas_openpcdet"))
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("../output/boreas_models/pointpillar/seed42_beam128/eval/epoch_80/val"),
    )
    parser.add_argument("--out", type=Path, default=Path("../output/boreas_summary/paper_pointcloud_figures"))
    parser.add_argument("--pdf-out", type=Path, default=Path("../output/pdf"))
    parser.add_argument("--suite", choices=("all", "modulo", "elevation", "phase16"), default="all")
    parser.add_argument("--split", default="val")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame-id")
    selection.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--boxes", choices=("gt", "prediction"), default="gt")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--xy-range", nargs=4, type=float, default=(-23, -23, 23, 23))
    parser.add_argument("--z-range", nargs=2, type=float, default=(-3, 5))
    parser.add_argument("--elevation", type=float, default=30)
    parser.add_argument("--azimuth", type=float, default=-55)
    parser.add_argument(
        "--view",
        choices=("3d", "bev", "both"),
        default="3d",
        help="Default: close oblique 3D paper panels",
    )
    parser.add_argument("--box-labels", action="store_true", help="Show class text next to boxes")
    args = parser.parse_args()

    bounds = np.array(
        [
            args.xy_range[0],
            args.xy_range[1],
            args.z_range[0],
            args.xy_range[2],
            args.xy_range[3],
            args.z_range[1],
        ],
        float,
    )
    fid = frame_id(args.data_root, args.split, args.frame_id, args.frame_index)
    common = gt(args.data_root, args.split, fid, bounds)
    outputs, metadata = [], {}
    views = ["bev", "3d"] if args.view == "both" else [args.view]
    suffix = args.view

    if args.suite in ("all", "modulo"):
        scenes = [scene(args, spec, fid, bounds, common) for spec in MOD]
        outputs.extend(render(args, scenes, fid, views, 4, f"paper_modulo_128_64_32_16_{suffix}.png"))
        metadata["modulo"] = scenes
    if args.suite in ("all", "elevation"):
        scenes = [scene(args, spec, fid, bounds, common) for spec in ELEV]
        outputs.extend(render(args, scenes, fid, views, 3, f"paper_elevation_{suffix}.png"))
        metadata["elevation"] = scenes
    if args.suite in ("all", "phase16"):
        scenes = [scene(args, spec, fid, bounds, common) for spec in PHASE]
        outputs.extend(render(args, scenes, fid, views, 4, f"paper_phase16_{suffix}.png"))
        metadata["phase16"] = scenes

    clean = {
        key: [
            {
                "condition": item["condition"],
                "shown_points": item["visible_points"],
                "total_points": item["input_points"],
                "boxes": len(item["boxes"]),
            }
            for item in value
        ]
        for key, value in metadata.items()
    }
    report = {
        "version": "paper_sampling_figures_v3",
        "frame_id": fid,
        "frame_index": args.frame_index,
        "boxes": args.boxes,
        "bounds": bounds.tolist(),
        "score_threshold": args.score_thresh,
        "view": args.view,
        "camera": {"elevation": args.elevation, "azimuth": args.azimuth},
        "files": [str(output.resolve()) for output in outputs],
        "conditions": clean,
        "note": "All panels share frame, crop, camera and point color mapping. GT mode is detector-independent.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paper_sampling_figures.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Frame:", fid)
    print("\n".join(report["files"]))


if __name__ == "__main__":
    main()
