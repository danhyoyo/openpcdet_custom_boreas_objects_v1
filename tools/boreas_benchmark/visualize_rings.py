"""Export same-frame Boreas detections at 128/64/32/16 rings as headless PNGs.

Uses result.pkl from test.py: no new inference, GPU or display server needed.
Run from tools; requires NumPy and Pillow.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BEAMS = (128, 64, 32, 16)
COLORS = {"Car": "#4ade80", "Pedestrian": "#fb923c", "Cyclist": "#c084fc"}
BACKGROUND = (12, 20, 32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_frame(data_root, split, frame_id=None, frame_index=0):
    reference = None
    for beam in BEAMS:
        ids = (data_root / str(beam) / "ImageSets" / f"{split}.txt").read_text().split()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate frame IDs at {beam} rings")
        if reference is not None and ids != reference:
            raise ValueError(f"{split} frame IDs differ at {beam} rings")
        reference = ids
    if frame_id is None:
        if not 0 <= frame_index < len(reference):
            raise ValueError(f"--frame-index must be in [0, {len(reference) - 1}]")
        frame_id = reference[frame_index]
    if frame_id not in reference or Path(frame_id).name != frame_id:
        raise ValueError(f"Frame {frame_id!r} is not in {split}")
    return frame_id


def load_scenes(eval_root, data_root, split, frame_id, frame_index, score_thresh, xy_range=None):
    frame_id = choose_frame(data_root, split, frame_id, frame_index)
    scenes = []
    reference = None
    manifests = []
    for beam in BEAMS:
        directory = eval_root / f"beam{beam}"
        manifest = json.loads((directory / "inference_manifest.json").read_text())
        # Compare actual weights and inference settings, not folder names.
        keys = ("checkpoint_sha256", "model_config_sha256", "precision", "classes",
                "point_cloud_range", "evaluation_seed", "score_cutoff",
                "nms_pre_maxsize", "nms_post_maxsize", "tf32_matmul", "tf32_cudnn")
        signature = {key: manifest.get(key) for key in keys}
        if not signature["checkpoint_sha256"] or not signature["point_cloud_range"]:
            raise ValueError(f"Incomplete inference manifest at {beam} rings; rerun test.py")
        if reference is not None and signature != reference:
            raise ValueError(f"Weights or inference settings differ at {beam} rings")
        reference = signature
        if Path(manifest["data_path"]).resolve() != (data_root / str(beam)).resolve():
            raise ValueError(f"Prediction data_path does not match {beam}-ring dataset")
        if score_thresh < float(manifest["score_cutoff"]):
            raise ValueError("--score-thresh is below the saved inference cutoff")
        prediction_path = directory / "result.pkl"
        if sha256(prediction_path) != manifest.get("prediction_sha256"):
            raise ValueError(f"Prediction hash differs at {beam} rings; rerun test.py")
        with prediction_path.open("rb") as stream:
            records = pickle.load(stream)
        matching = [r for r in records if str(r["frame_id"]) == frame_id]
        if len(matching) != 1:
            raise ValueError(f"Expected one prediction for frame {frame_id} at {beam} rings")
        record = matching[0]
        boxes = np.asarray(record["boxes_lidar"])
        names = np.asarray(record["name"])
        scores = np.asarray(record["score"])
        if boxes.ndim != 2 or boxes.shape[1] < 7 or len(boxes) != len(scores) or len(names) != len(scores):
            raise ValueError(f"Malformed prediction arrays at {beam} rings")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f"Non-finite prediction at {beam} rings")
        bounds = np.array(manifest["point_cloud_range"], dtype=float)
        if xy_range is not None:
            bounds[[0, 1, 3, 4]] = xy_range
        points = np.load(data_root / str(beam) / "points" / f"{frame_id}.npy")
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Malformed point cloud at {beam} rings")
        visible_points = np.isfinite(points[:, :3]).all(axis=1)
        visible_points &= ((points[:, :3] >= bounds[:3]) & (points[:, :3] <= bounds[3:])).all(axis=1)
        keep = (scores >= score_thresh)
        keep &= ((boxes[:, :3] >= bounds[:3]) & (boxes[:, :3] <= bounds[3:])).all(axis=1)
        scenes.append({"beam": beam, "points": points[visible_points, :3],
                       "input_points": len(points), "boxes": boxes[keep, :7],
                       "names": names[keep], "scores": scores[keep], "bounds": bounds})
        manifests.append({"beam": beam, "prediction": str(prediction_path.resolve()),
                          "prediction_sha256": manifest["prediction_sha256"],
                          "visible_points": int(visible_points.sum()), "input_points": len(points),
                          "detections": int(keep.sum())})
    return frame_id, scenes, reference, manifests


def box_corners(box):
    # First four corners bottom face, next four top face; LiDAR z is box center.
    local = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    local *= np.asarray(box[3:6]) / 2
    c, s = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def projection(bounds, view, size, elevation=35, azimuth=-55, paper_style=False):
    if view == "bev":
        basis = np.array([[1, 0], [0, 1], [0, 0]])
    else:
        az, el = np.deg2rad([azimuth, elevation])
        basis = np.array([[-np.sin(az), -np.cos(az) * np.sin(el)],
                          [np.cos(az), -np.sin(az) * np.sin(el)], [0, np.cos(el)]])
    corners = np.array(list(itertools.product(*zip(bounds[:3], bounds[3:]))))
    projected = corners @ basis
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    width, height = size
    if paper_style:
        horizontal_margin, vertical_margin, vertical_shift = -20, -10, 0
    else:
        horizontal_margin, vertical_margin, vertical_shift = 130, 215, 30
    scale = min((width - horizontal_margin) / (upper[0] - lower[0]),
                (height - vertical_margin) / (upper[1] - lower[1]))
    if paper_style:
        # Fill the panel like a close paper crop; peripheral returns may meet
        # or clip at the image edge while the camera and scene center stay fixed.
        scale *= 1.4
    center = (upper + lower) / 2
    def project(xyz):
        uv = (np.asarray(xyz) @ basis - center) * scale
        return uv * [1, -1] + [width / 2, height / 2 + vertical_shift]
    return project


def font(size):
    candidates = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "C:/Windows/Fonts/arial.ttf")
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def render_panel(scene, frame_id, view, score_thresh, size=(1200, 1050),
                 elevation=35, azimuth=-55, show_labels=True,
                 paper_style=False, title=None):
    width, height = size
    bounds = scene["bounds"]
    project = projection(bounds, view, size, elevation, azimuth, paper_style)
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    # Grid is shared across rings and uses LiDAR coordinates in meters.
    for x in np.arange(np.ceil(bounds[0] / 10) * 10, bounds[3] + 0.01, 10):
        line = project([[x, bounds[1], bounds[2]], [x, bounds[4], bounds[2]]])
        draw.line([tuple(p) for p in line], fill=(29, 42, 57), width=1)
    for y in np.arange(np.ceil(bounds[1] / 10) * 10, bounds[4] + 0.01, 10):
        line = project([[bounds[0], y, bounds[2]], [bounds[3], y, bounds[2]]])
        draw.line([tuple(p) for p in line], fill=(29, 42, 57), width=1)
    points = scene["points"]
    # Render all visible raw points, before voxel caps. No random display thinning.
    pixels = np.rint(project(points)).astype(int)
    z = np.clip((points[:, 2] - bounds[2]) / (bounds[5] - bounds[2]), 0, 1)
    colors = np.column_stack((75 + 150 * z, 130 + 100 * z, 210 - 85 * z)).astype(np.uint8)
    canvas = np.array(image)
    order = np.argsort(z)
    pixels, colors = pixels[order], colors[order]
    for dx, dy in ((0, 0), (1, 0), (0, 1)):
        xx, yy = pixels[:, 0] + dx, pixels[:, 1] + dy
        top_limit = 0 if paper_style else 135
        bottom_limit = height if paper_style else height - 65
        visible = (xx >= 0) & (xx < width) & (yy >= top_limit) & (yy < bottom_limit)
        canvas[yy[visible], xx[visible]] = colors[visible]
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    if view == "3d":
        edges += [(4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for box, name, score in zip(scene["boxes"], scene["names"], scene["scores"]):
        color = COLORS.get(str(name), "#facc15")
        corners = project(box_corners(box))
        for start, end in edges:
            draw.line([tuple(corners[start]), tuple(corners[end])], fill=color, width=3)
        center = project([box[:3]])[0]
        front = box[:3] + [np.cos(box[6]) * box[3] / 2, np.sin(box[6]) * box[3] / 2, 0]
        draw.line([tuple(center), tuple(project([front])[0])], fill=color, width=3)
        if show_labels:
            position = (int(corners[:, 0].min()), int(corners[:, 1].min()) - 19)
            label = f"{name} {score:.2f}" if scene.get("show_scores", True) else str(name)
            rectangle = draw.textbbox(position, label, font=font(15))
            draw.rectangle(rectangle, fill=BACKGROUND)
            draw.text(position, label, fill=color, font=font(15))
    origin = project([[0, 0, 0]])[0]
    draw.ellipse((origin[0]-5, origin[1]-5, origin[0]+5, origin[1]+5), fill="white")
    if not paper_style:
        for endpoint, color, label in (([10, 0, 0], "#f87171", "+X"), ([0, 10, 0], "#60a5fa", "+Y")):
            tip = project([endpoint])[0]
            draw.line([tuple(origin), tuple(tip)], fill=color, width=2)
            draw.text(tuple(tip), label, fill=color, font=font(17))
    if paper_style:
        return image
    # Solid header/footer keep large scenes and box labels clear of metadata.
    draw.rectangle((0, 0, width, 125), fill=BACKGROUND)
    draw.rectangle((0, height-60, width, height), fill=BACKGROUND)
    draw.text((30, 18), f"{scene['beam']} rings  |  {'Top view (BEV)' if view == 'bev' else '3D view'}",
              fill="white", font=font(30))
    draw.text((30, 62), f"Frame {frame_id}", fill="#a8b5c5", font=font(20))
    box_kind = scene.get("box_kind", "detections")
    score_note = f"  |  score >= {score_thresh:g}" if scene.get("show_scores", True) else ""
    draw.text((30, 93), f"{len(points):,} points in view  |  {len(scene['boxes'])} {box_kind}{score_note}",
              fill="#a8b5c5", font=font(18))
    x = 30
    for name, color in COLORS.items():
        draw.rectangle((x, height-38, x+14, height-24), fill=color)
        draw.text((x+23, height-42), name, fill=color, font=font(18))
        x += 190
    draw.text((x, height-42), "Grid: 10 m | points colored by height", fill="#a8b5c5", font=font(17))
    return image


def export(args):
    if not 0 <= args.score_thresh <= 1:
        raise ValueError("--score-thresh must be in [0, 1]")
    if args.xy_range and not (args.xy_range[0] < args.xy_range[2] and args.xy_range[1] < args.xy_range[3]):
        raise ValueError("--xy-range needs xmin ymin xmax ymax with increasing bounds")
    frame_id, scenes, signature, sources = load_scenes(
        args.eval_root, args.data_root, args.split, args.frame_id, args.frame_index,
        args.score_thresh, args.xy_range)
    output = (args.out or args.eval_root / "visualizations") / frame_id
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for view in ("bev", "3d"):
        combined = Image.new("RGB", (2400, 2100), BACKGROUND)
        for index, scene in enumerate(scenes):
            panel = render_panel(scene, frame_id, view, args.score_thresh,
                                 elevation=args.elevation, azimuth=args.azimuth,
                                 show_labels=not args.no_labels)
            target = output / f"ring{scene['beam']}_{view}.png"
            panel.save(target)
            files.append(str(target.resolve()))
            combined.paste(panel, ((index % 2) * 1200, (index // 2) * 1050))
        target = output / f"comparison_{view}.png"
        combined.save(target)
        files.append(str(target.resolve()))
    metadata = {"frame_id": frame_id, "split": args.split, "score_threshold": args.score_thresh,
                "point_cloud_range": scenes[0]["bounds"].tolist(),
                "camera": {"elevation": args.elevation, "azimuth": args.azimuth},
                "inference": signature, "sources": sources, "files": files,
                "note": "Saved test detections; display cutoff only. Raw points before voxel caps; no display subsampling. Views use a common orthographic camera."}
    (output / "visualization.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Frame: {frame_id}\nImages: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path,
                        help="epoch_N/val folder containing beam128/beam64/beam32/beam16")
    parser.add_argument("--data-root", type=Path, default=Path("/home/ducanh/datasets/boreas_openpcdet"))
    parser.add_argument("--split", default="val")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame-id", help="Exact shared frame ID")
    selection.add_argument("--frame-index", type=int, default=0, help="Shared split index, default 0")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Display cutoff only, default 0.3")
    parser.add_argument("--xy-range", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--elevation", type=float, default=35)
    parser.add_argument("--azimuth", type=float, default=-55)
    parser.add_argument("--no-labels", action="store_true", help="Hide box text for crowded scenes")
    parser.add_argument("--out", type=Path, help="Output parent folder; appends frame ID")
    args = parser.parse_args()
    try:
        export(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"Cannot export comparison: {error}\n")


if __name__ == "__main__":
    main()
