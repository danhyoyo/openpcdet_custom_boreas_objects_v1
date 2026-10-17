"""Persist accuracy metrics and optional local radial-distance evaluation."""
import json
from pathlib import Path
import numpy as np
from boreas_benchmark.evaluate import evaluate


def write_reports(dataset, result_dir, metrics, range_eval, logger, cfg=None, args=None):
    result_dir = Path(result_dir)
    if cfg is not None and args is not None:
        from boreas_benchmark.extended import sha256_file
        import torch
        post = cfg.MODEL.DENSE_HEAD.get('POST_PROCESSING', cfg.MODEL.POST_PROCESSING)
        nms = post.get('NMS_CONFIG', {})
        model_json = json.dumps(cfg.MODEL, sort_keys=True, default=str)
        import hashlib
        checkpoint = getattr(args, 'ckpt', None)
        manifest = {
            'version': 'inference_manifest_v1', 'precision': 'FP32', 'batch_size': args.batch_size,
            'data_path': str(dataset.root_path.resolve()),
            'prediction_sha256': sha256_file(result_dir / 'result.pkl'),
            'model_config_sha256': hashlib.sha256(model_json.encode()).hexdigest(),
            'score_cutoff': post.get('SCORE_THRESH'),
            'nms_pre_maxsize': nms.get('NMS_PRE_MAXSIZE'),
            'nms_post_maxsize': nms.get('NMS_POST_MAXSIZE'),
            'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
            'checkpoint_sha256': sha256_file(checkpoint) if checkpoint else None,
            'evaluation_seed': getattr(args, 'seed', None),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'classes': list(cfg.CLASS_NAMES), 'point_cloud_range': list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
        }
        (result_dir / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    serializable = {key: float(value) for key, value in metrics.items()
                    if isinstance(value, (int, float, np.number))}
    classes = {}
    for name in dataset.class_names:
        key = name + "_3d/moderate_R40"
        if key in serializable:
            classes[name] = serializable[key]
    accuracy = {"protocol": "KITTI evaluator 3D AP_R40; supplementary only: synthetic 2D boxes, no real difficulty tiers",
                "classes": classes, "mAP_R40": float(np.mean(list(classes.values())))
                if len(classes) == len(dataset.class_names) else None,
                "metrics": serializable}
    (result_dir / "kitti_accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    if not range_eval:
        accuracy = {"protocol": "Local AP40 not requested; rerun evaluation with --range_eval",
                    "classes": {}, "mAP40": None, "supplementary_file": "kitti_accuracy.json"}
    if range_eval:
        infos = dataset.root_path / ("custom_infos_" + dataset.split + ".pkl")
        report = evaluate(result_dir / "result.pkl", infos)
        from boreas_benchmark.extended import sha256_file, truth_fingerprint
        import pickle
        report['prediction_sha256'] = sha256_file(result_dir / 'result.pkl')
        with infos.open('rb') as stream:
            report['truth_sha256'] = truth_fingerprint(pickle.load(stream))
        (result_dir / "range_ap.json").write_text(json.dumps(report, indent=2) + "\n")
        accuracy = {"protocol": report["metric"], "protocol_version": report["protocol_version"],
                    "classes": {name: entry["AP40"] for name, entry in report["ranges"]["all"]["classes"].items()},
                    "mAP40": report["ranges"]["all"]["mAP40"],
                    "iou_thresholds": report["iou_thresholds"], "ignore_policy": report["ignore_policy"]}
        for label, values in report["ranges"].items():
            logger.info("Local AP40 %s: %s", label, values)
    (result_dir / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
