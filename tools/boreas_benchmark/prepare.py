"""Download and prepare Boreas-Objects-V1 for OpenPCDet's CustomDataset.

Run in WSL. Prepare needs numpy and asrl-pyboreas; optional download needs boto3.
The official object-detection test labels are withheld, so the prepared
validation split is carved from official training windows.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

import numpy as np

CLASSES = ("Car", "Pedestrian", "Cyclist", "Misc")
ROOT = Path("/home/ducanh/datasets")


def download(destination: Path, plan_only: bool) -> None:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefixes = (
        "boreas-objects-v1/lidar/",
        "boreas-objects-v1/labels_detection/",
    )
    candidates = {}
    for prefix in prefixes:
        pages = client.get_paginator("list_objects_v2").paginate(Bucket="boreas", Prefix=prefix)
        for page in pages:
            candidates.update({obj["Key"]: obj["Size"] for obj in page.get("Contents", [])
                               if not obj["Key"].endswith("/")})
    label_keys = {key: size for key, size in candidates.items()
                  if "/labels_detection/" in key and key.endswith(".txt")}
    if not label_keys:
        raise RuntimeError("No detection labels listed; check S3 access")
    sizes = dict(label_keys)
    for key in label_keys:
        lidar_key = key.replace("/labels_detection/", "/lidar/").removesuffix(".txt") + ".bin"
        if lidar_key not in candidates:
            raise RuntimeError(f"Missing LiDAR scan for {key}")
        sizes[lidar_key] = candidates[lidar_key]
    total = sum(sizes.values())
    print(json.dumps({"files": len(sizes), "bytes": total, "GiB": round(total / 2**30, 2)}, indent=2))
    if plan_only:
        return
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    missing = sum(size for key, size in sizes.items() if not (destination / key).is_file()
                  or (destination / key).stat().st_size != size)
    if free < missing:
        raise RuntimeError(f"Need {missing / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free")
    pending = [(key, size) for key, size in sizes.items()
               if not (destination / key).is_file() or (destination / key).stat().st_size != size]
    transfer = TransferConfig(max_concurrency=2)

    def fetch(entry: tuple[str, int]) -> None:
        key, size = entry
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        client.download_file("boreas", key, str(temporary), Config=transfer)
        if temporary.stat().st_size != size:
            raise RuntimeError(f"Incomplete download: {key}")
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, future in enumerate(as_completed(pool.submit(fetch, item) for item in pending), 1):
            future.result()
            if index % 100 == 0 or index == len(pending):
                print(f"Downloaded {index}/{len(pending)} files", flush=True)
    for key, size in sizes.items():
        target = destination / key
        if not target.is_file() or target.stat().st_size != size:
            raise RuntimeError(f"Missing or incomplete download: {key}")


def boxes_from_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    boxes, names = [], []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 9 or fields[1] not in CLASSES:
            raise ValueError(f"Invalid label at {path}:{lineno}")
        box = [float(x) for x in fields[5:8] + fields[2:5] + fields[8:9]]
        if not np.isfinite(box).all() or min(box[3:6]) <= 0:
            raise ValueError(f"Invalid box at {path}:{lineno}")
        if not (-75 <= box[0] < 75 and -75 <= box[1] < 75 and -3 <= box[2] < 5):
            continue
        boxes.append(box)
        names.append(fields[1])
    return np.asarray(boxes, np.float32).reshape(-1, 7), np.asarray(names, dtype="<U10")


def beam_subset(points: np.ndarray, count: int, phase: int = 0) -> np.ndarray:
    if count not in (128, 64, 32, 16):
        raise ValueError("Beam count must be 128, 64, 32, or 16")
    modulus = 128 // count
    if not 0 <= phase < modulus:
        raise ValueError(f"Phase for {count} beams must be in [0, {modulus - 1}]")
    if count == 128:
        return points
    ring = points[:, 4]
    rounded = np.rint(ring)
    if not np.isfinite(ring).all() or not np.all(np.abs(ring - rounded) < 0.01):
        raise ValueError("Nonintegral LiDAR ring IDs")
    if np.any((rounded < 0) | (rounded > 127)):
        raise ValueError("LiDAR ring ID outside [0, 127]")
    return points[(rounded.astype(np.int16) % modulus) == phase]


def prepare(source: Path, destination: Path, beams: int, maximum: int,
            phase: int = 0) -> None:
    # Import only the split definition: importing pyboreas itself pulls in
    # plotting extensions that may not match the local NumPy ABI.
    from importlib import metadata, util

    split_path = metadata.distribution("asrl-pyboreas").locate_file("pyboreas/data/splits.py")
    spec = util.spec_from_file_location("boreas_official_splits", split_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official splits at {split_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj_train = module.obj_train

    modulus = 128 // beams if beams in (128, 64, 32, 16) else 0
    if beams not in (128, 64, 32, 16) or not 0 <= phase < modulus:
        raise ValueError("Invalid beam count or modulo phase")
    raw = source / "boreas-objects-v1"
    labels_dir, lidar_dir = raw / "labels_detection", raw / "lidar"
    if not labels_dir.is_dir() or not lidar_dir.is_dir():
        raise FileNotFoundError("Download lidar and labels_detection first")
    windows = [(int(a), int(b)) for _, a, b in obj_train]
    validation = [window for index, window in enumerate(windows) if index % 5 == 0]
    training = [window for index, window in enumerate(windows) if index % 5 != 0]
    gap = 2_000_000  # Purge training frames within two seconds of validation.

    def split_of(timestamp: int) -> str | None:
        if any(a <= timestamp <= b for a, b in validation):
            return "val"
        if any(a <= timestamp <= b for a, b in training):
            if any(a - gap <= timestamp <= b + gap for a, b in validation):
                return None
            return "train"
        return None

    condition = str(beams) if phase == 0 else f"{beams}_phase{phase}"
    out = destination / condition
    for folder in ("points", "labels", "ImageSets"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    ids = {"train": [], "val": []}
    infos = {"train": [], "val": []}
    counts = Counter()
    range_counts = Counter()
    point_counts = []
    for label in sorted(labels_dir.glob("*.txt")):
        if not label.stem.isdecimal():
            continue
        split = split_of(int(label.stem))
        if split is None:
            continue
        lidar = lidar_dir / (label.stem + ".bin")
        if not lidar.is_file():
            raise FileNotFoundError(lidar)
        data = np.fromfile(lidar, dtype=np.float32)
        if data.size % 6:
            raise ValueError(f"Bad 6-field LiDAR record: {lidar}")
        points = beam_subset(data.reshape(-1, 6), beams, phase)
        point_counts.append(len(points))
        boxes, names = boxes_from_label(label)
        counts.update(names.tolist())
        for box, name in zip(boxes, names):
            distance = float(np.hypot(box[0], box[1]))
            bucket = "0-20m" if distance < 20 else "20-40m" if distance < 40 else "40-60m" if distance < 60 else "60m+"
            range_counts[f"{split}:{name}:{bucket}"] += 1
        np.save(out / "points" / (label.stem + ".npy"), points[:, :4])
        with (out / "labels" / (label.stem + ".txt")).open("w") as handle:
            for box, name in zip(boxes, names):
                handle.write(" ".join(map(str, box.tolist())) + f" {name}\n")
        if sum(map(len, ids.values())) % 100 == 0:
            print(f"Preparing {beams} beams, phase {phase}: {sum(map(len, ids.values()))} frames", flush=True)
        ids[split].append(label.stem)
        infos[split].append({
            "point_cloud": {"num_features": 4, "lidar_idx": label.stem},
            "annos": {"name": names, "gt_boxes_lidar": boxes},
        })
        if maximum and sum(map(len, ids.values())) >= maximum:
            break
    if not ids["train"] or not ids["val"]:
        raise ValueError("Both train and val need labelled scans")
    for split in ("train", "val"):
        (out / "ImageSets" / (split + ".txt")).write_text("\n".join(ids[split]) + "\n")
        with (out / ("custom_infos_" + split + ".pkl")).open("wb") as handle:
            pickle.dump(infos[split], handle, protocol=4)
    manifest = {
        "source": str(raw), "beam_count": beams,
        "sampling_strategy": "ring_id_modulo",
        "modulo": modulus, "phase": phase,
        "retained_ring_ids": list(range(phase, 128, modulus)),
        "train_frames": len(ids["train"]), "val_frames": len(ids["val"]),
        "class_boxes": dict(counts), "purge_gap_seconds": 2,
        "split_class_range_boxes": dict(range_counts),
        "median_points_per_scan": int(np.median(point_counts)),
        "validation_window_rule": "every fifth official obj_train interval",
        "smoke_test_only": bool(maximum),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("download")
    get.add_argument("--destination", type=Path, default=ROOT / "boreas")
    get.add_argument("--plan", action="store_true")
    make = sub.add_parser("prepare")
    make.add_argument("--source", type=Path, default=ROOT / "boreas")
    make.add_argument("--destination", type=Path, default=ROOT / "boreas_openpcdet")
    make.add_argument("--beams", type=int, default=128)
    make.add_argument("--phase", type=int, default=0,
                      help="Modulo phase in [0, 128/beams); phase 0 preserves legacy paths")
    make.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.command == "download":
        download(args.destination, args.plan)
    else:
        prepare(args.source, args.destination, args.beams, args.max_frames, args.phase)


if __name__ == "__main__":
    main()
