"""Rotated overlap for KITTI evaluation using OpenPCDet CUDA ops.

The public interface and clockwise box convention match the original Numba
implementation. AP matching and interpolation remain in eval.py.
"""
import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_cuda


@torch.no_grad()
def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Compute overlap for [x, y, dx, dy, clockwise_angle] boxes.

    criterion: -1 is IoU, 0 divides by boxes area, 1 by query area,
    and any other value returns intersection area.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    if boxes.ndim != 2 or query_boxes.ndim != 2 or boxes.shape[1] != 5 or query_boxes.shape[1] != 5:
        raise ValueError("Rotated boxes must have shape (N, 5)")
    n, k = len(boxes), len(query_boxes)
    if not n or not k:
        return np.zeros((n, k), dtype=np.float32)
    device = torch.device("cuda", device_id)
    with torch.cuda.device(device):
        def convert(array):
            data = torch.as_tensor(array, device=device)
            out = torch.zeros((len(array), 7), dtype=torch.float32, device=device)
            out[:, :2] = data[:, :2]
            out[:, 3:5] = data[:, 2:4]
            out[:, 5] = 1.0
            # KITTI evaluator uses clockwise angles; CUDA ops use counterclockwise.
            out[:, 6] = -data[:, 4]
            return out.contiguous()

        first, second = convert(boxes), convert(query_boxes)
        overlap = torch.zeros((n, k), dtype=torch.float32, device=device)
        iou3d_nms_cuda.boxes_overlap_bev_gpu(first, second, overlap)
        area_first = (first[:, 3] * first[:, 4])[:, None]
        area_second = (second[:, 3] * second[:, 4])[None, :]
        if criterion == -1:
            overlap = overlap / (area_first + area_second - overlap).clamp_min(1e-8)
        elif criterion == 0:
            overlap = overlap / area_first.clamp_min(1e-8)
        elif criterion == 1:
            overlap = overlap / area_second.clamp_min(1e-8)
        return overlap.cpu().numpy()
