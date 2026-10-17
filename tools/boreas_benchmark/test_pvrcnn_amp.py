"""CUDA regression tests plus an optional real-data PV-RCNN AMP smoke run.

Run from tools: python boreas_benchmark/test_pvrcnn_amp.py
Optional: python boreas_benchmark/test_pvrcnn_amp.py --model-smoke --batch-size 2
The smoke run neither saves checkpoints nor changes existing experiment folders.
"""
import argparse
import copy
import sys
import unittest
from pathlib import Path

# Avoid the neighboring profile.py shadowing Python's standard-library module.
sys.path[:] = [entry for entry in sys.path
               if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import _init_path
import torch
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as ops
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG

@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PointNetAMPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_grouping_matches_reference_and_returns_gradients_to_half_inputs(self):
        features = torch.arange(21, device="cuda", dtype=torch.float16).reshape(7, 3).requires_grad_()
        counts = torch.tensor([4, 3], device="cuda", dtype=torch.int32)
        idx = torch.tensor([[0, 1], [2, 3], [0, 2]], device="cuda", dtype=torch.int32)
        query_counts = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
        reference_input = features.detach().clone().requires_grad_()
        absolute_idx = idx.long() + torch.tensor([[0], [0], [4]], device="cuda")
        reference = reference_input.float()[absolute_idx].permute(0, 2, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = ops.grouping_operation(features, counts, idx, query_counts)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference)
        actual.square().mean().backward()
        reference.square().mean().backward()
        self.assertEqual(features.grad.dtype, torch.float16)
        torch.testing.assert_close(features.grad, reference_input.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        # The FP32 path must also continue working outside autocast.
        fp32 = features.detach().float().requires_grad_()
        result = ops.grouping_operation(fp32, counts, idx, query_counts)
        torch.testing.assert_close(result, reference.detach())
        result.square().mean().backward()
        torch.testing.assert_close(fp32.grad, reference_input.grad.float(), atol=.002, rtol=.002)

    def test_half_coordinates_ball_query_and_empty_neighborhood(self):
        xyz = torch.tensor([[0,0,0], [.1,0,0], [2,0,0]], device="cuda", dtype=torch.float16)
        centers = torch.tensor([[0,0,0], [10,0,0]], device="cuda", dtype=torch.float16)
        count = torch.tensor([3], device="cuda", dtype=torch.int32)
        new_count = torch.tensor([2], device="cuda", dtype=torch.int32)
        with torch.autocast("cuda", dtype=torch.float16):
            idx, empty = ops.ball_query(.5, 2, xyz, count, centers, new_count)
        self.assertEqual(idx.dtype, torch.int32)
        self.assertEqual(empty.tolist(), [False, True])
        self.assertEqual(idx[0].tolist(), [0, 1])

    def test_pointnet_mlps_remain_half_and_backward_is_finite(self):
        xyz = torch.randn(16, 3, device="cuda")
        counts = torch.tensor([8,8], device="cuda", dtype=torch.int32)
        query_counts = torch.tensor([2,2], device="cuda", dtype=torch.int32)
        centers = xyz[[0,1,8,9]].contiguous()
        source = torch.randn(16, 4, device="cuda", requires_grad=True)
        projection = torch.nn.Linear(4, 8).cuda()
        module = StackSAModuleMSG(radii=[2.], nsamples=[4], mlps=[[8,8,8]]).cuda().train()
        with torch.autocast("cuda", dtype=torch.float16):
            features = projection(source)
            self.assertEqual(features.dtype, torch.float16)
            _, result = module(xyz, counts, centers, query_counts, features)
            self.assertEqual(result.dtype, torch.float16)
            loss = result.float().square().mean()
        loss.backward()
        for parameter in [source] + list(projection.parameters()) + list(module.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


    def test_roi_bce_loss_matches_fp32_and_masks_ignored_labels(self):
        from types import SimpleNamespace
        from pcdet.models.roi_heads.roi_head_template import RoIHeadTemplate
        head = SimpleNamespace(model_cfg=SimpleNamespace(LOSS_CONFIG=SimpleNamespace(
            CLS_LOSS="BinaryCrossEntropy", LOSS_WEIGHTS={"rcnn_cls_weight": 1.})))
        for dtype in (torch.float32, torch.float16):
            logits = torch.tensor([-2., 0., 2., 1.], device="cuda", dtype=dtype, requires_grad=True)
            labels = torch.tensor([0., .3, 1., -1.], device="cuda")
            with torch.autocast("cuda", enabled=dtype == torch.float16):
                loss, _ = RoIHeadTemplate.get_box_cls_layer_loss(head, {
                    "rcnn_cls": logits, "rcnn_cls_labels": labels})
            reference = torch.nn.functional.binary_cross_entropy(logits[:3].float().sigmoid(), labels[:3])
            torch.testing.assert_close(loss, reference)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad[-1].item(), 0)
        # Extreme half logits must stay finite without an unsafe sigmoid+BCE.
        logits = torch.tensor([-100., 100.], device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.autocast("cuda"):
            loss, _ = RoIHeadTemplate.get_box_cls_layer_loss(head, {
                "rcnn_cls": logits, "rcnn_cls_labels": torch.tensor([1., 0.], device="cuda")})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


def model_smoke(batch_size, steps):
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.custom.custom_dataset import CustomDataset
    from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
    from pcdet.utils import common_utils
    from train_utils.optimization import build_optimizer
    common_utils.set_random_seed(42)
    cfg_from_yaml_file("cfgs/boreas_models/pv_rcnn.yaml", cfg)
    logger = common_utils.create_logger()
    dataset = CustomDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=True, logger=logger)
    samples = [dataset[i] for i in range(batch_size)]
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset).cuda().train()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    # A modest test loss scale avoids spending smoke steps on initial scale backoff.
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    model_func = model_fn_decorator()
    updated = False
    for i in range(steps):
        batch = dataset.collate_batch(copy.deepcopy(samples))
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _, _ = model_func(model, batch)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite model loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = all(torch.isfinite(g).all().item() for g in grads)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated |= finite
        print(f"AMP train step {i+1}: loss={loss.item():.6f}, finite_gradients={finite}, scale={scale}", flush=True)
    if not updated:
        raise AssertionError("No finite-gradient optimizer step completed")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        predictions, _ = model(batch)
    for pred in predictions:
        assert torch.isfinite(pred["pred_boxes"]).all()
        assert torch.isfinite(pred["pred_scores"]).all()
    torch.cuda.synchronize()
    print(f"PV-RCNN real-data AMP train/backward/optimizer/inference passed, batch={batch_size}; peak GPU={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PointNetAMPTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.model_smoke:
        if args.batch_size < 1 or args.steps < 1:
            raise ValueError("batch-size and steps must be positive")
        model_smoke(args.batch_size, args.steps)
