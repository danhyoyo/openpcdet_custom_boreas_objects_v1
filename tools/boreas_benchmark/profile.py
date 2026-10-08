"""Batch-1 latency and CUDA-memory profile for a trained Boreas detector.

Run from OpenPCDet/tools. Model-forward time includes its postprocessing/NMS.
End-to-end time includes loading a prepared .npy frame, CPU preprocessing,
host-to-GPU transfer, model forward, and postprocessing; it excludes camera
capture, raw Boreas .bin decoding, rendering, and result serialization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Prevent this file from shadowing the standard-library profile module.
sys.path[:] = [entry for entry in sys.path
               if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: E402,F401

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.custom.custom_dataset import CustomDataset  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def describe(values: list[float]) -> dict:
    array = np.asarray(values)
    return {"mean_ms": round(float(array.mean()), 3),
            "p50_ms": round(float(np.percentile(array, 50)), 3),
            "p95_ms": round(float(np.percentile(array, 95)), 3)}


def profile(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for latency profiling")
    if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':
        raise RuntimeError('Unset CUDA_LAUNCH_BLOCKING before latency profiling')
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError('--frames must be positive and --warmup nonnegative')
    common_utils.set_random_seed(args.seed)
    cfg_from_yaml_file(str(args.cfg), cfg)
    if args.boreas_beams is not None:
        from boreas_benchmark.beam_config import select_beams
        select_beams(cfg, args.boreas_beams)
    logger = common_utils.create_logger()
    dataset = CustomDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
                            logger=logger)
    if not len(dataset):
        raise RuntimeError("Validation infos are empty; run prepare.py first")
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
    model.cuda().eval()
    indices = np.linspace(0, len(dataset) - 1, min(args.frames, len(dataset)),
                          dtype=int).tolist()
    if not indices:
        raise RuntimeError("No validation frames selected")
    warmup = indices[:min(args.warmup, len(indices))]

    def run_one(index: int):
        torch.cuda.synchronize()
        start = time.perf_counter()
        sample = dataset[index]
        batch = dataset.collate_batch([sample])
        preprocess_end = time.perf_counter()
        voxel_count = int(len(batch.get("voxel_coords", [])))
        load_data_to_gpu(batch)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", enabled=args.amp):
            model(batch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return ((preprocess_end - start) * 1000,
                (forward_start - preprocess_end) * 1000,
                (end - forward_start) * 1000,
                voxel_count)

    for index in warmup:
        run_one(index)
    pre, transfer, forward, end_to_end, voxels, peaks, reserves = [], [], [], [], [], [], []
    for index in indices:
        torch.cuda.reset_peak_memory_stats()
        p, t, f, count = run_one(index)
        pre.append(p)
        transfer.append(t)
        forward.append(f)
        end_to_end.append(p + t + f)
        voxels.append(count)
        peaks.append(torch.cuda.max_memory_allocated() / 2**20)
        reserves.append(torch.cuda.max_memory_reserved() / 2**20)
    result = {
        "model": args.cfg.stem, "checkpoint": str(args.ckpt),
        "frames": len(indices), "batch_size": 1,
        "precision": "AMP" if args.amp else "FP32",
        "seed": args.seed, "data_path": str(dataset.root_path),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "warmup_frames": len(warmup),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "preprocess_including_npy_io": describe(pre),
        "host_to_gpu": describe(transfer),
        "model_forward_including_postprocess": describe(forward),
        "timing_note": "Synchronized CPU wall clock; full forward includes all head decoding and NMS. No architecture-dependent network/NMS split.",
        "end_to_end": describe(end_to_end),
        "fps_from_mean_end_to_end": round(1000 / float(np.mean(end_to_end)), 3),
        "peak_cuda_allocated_mib": round(max(peaks), 2),
        "peak_cuda_reserved_mib": round(max(reserves), 2),
        "active_voxels_median": int(np.median(voxels)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Optional AMP inference; default is FP32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boreas_beams", type=int, choices=[128, 64, 32, 16])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = profile(args)
    content = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
