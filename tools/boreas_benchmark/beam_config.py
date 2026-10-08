"""Select prepared beam data and validate it before training/inference."""
import json
from pathlib import Path


def select_beams(cfg, beams):
    if cfg.DATA_CONFIG.DATASET != "CustomDataset":
        raise ValueError("--boreas_beams requires the Boreas CustomDataset config")
    current = Path(cfg.DATA_CONFIG.DATA_PATH)
    root = current.parent if current.name in ("128", "64", "32", "16") else current
    target = root / str(beams)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare {beams}-beam data first: python boreas_benchmark/prepare.py prepare --beams {beams} --destination {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("beam_count") != beams or manifest.get("smoke_test_only"):
        raise ValueError("Beam dataset is inconsistent or contains only smoke-test frames")
    for split in ("train", "val"):
        if not (target / f"custom_infos_{split}.pkl").is_file():
            raise FileNotFoundError(target / f"custom_infos_{split}.pkl")
    baseline = root / "128" / "ImageSets"
    if baseline.is_dir():
        for split in ("train", "val"):
            if (baseline / f"{split}.txt").read_text().split() != (target / "ImageSets" / f"{split}.txt").read_text().split():
                raise ValueError(f"{split} frame IDs differ from the 128-beam baseline")
    cfg.DATA_CONFIG.DATA_PATH = str(target)
    cfg.DATA_CONFIG.BOREAS_BEAMS = beams
