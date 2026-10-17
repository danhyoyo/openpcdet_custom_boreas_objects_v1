"""Boreas dataset variant that mixes precomputed modulo phases during training."""

from pathlib import Path

import numpy as np

from .custom_dataset import CustomDataset


class BoreasPhaseMixDataset(CustomDataset):
    """Load the same labelled frame from a sampled beam/phase condition.

    Phase mixing is active only for training. Validation always reads the
    condition selected by DATA_PATH (or ``--boreas_beams``), so evaluation
    remains deterministic and compatible with the existing benchmark scripts.
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None,
                 logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        mix_cfg = dataset_cfg.get('PHASE_MIX', {})
        self.phase_mix_enabled = bool(training and mix_cfg.get('ENABLED', False))
        self.phase_roots = []
        self.phase_probabilities = None
        if not self.phase_mix_enabled:
            return

        conditions = mix_cfg.get('CONDITIONS', [])
        if not conditions:
            raise ValueError('PHASE_MIX.ENABLED requires at least one condition')
        dataset_root = Path(self.root_path).parent
        weights = []
        for condition in conditions:
            name = str(condition.NAME)
            condition_root = dataset_root / name
            points_dir = condition_root / 'points'
            split_file = condition_root / 'ImageSets' / 'train.txt'
            if not points_dir.is_dir() or not split_file.is_file():
                raise FileNotFoundError(
                    f'PR-DSVT phase condition is not prepared: {condition_root}'
                )
            condition_ids = split_file.read_text().split()
            reference_ids = (Path(self.root_path) / 'ImageSets' / 'train.txt').read_text().split()
            if condition_ids != reference_ids:
                raise ValueError(f'Train frame IDs differ for phase condition {name}')
            self.phase_roots.append(condition_root)
            weights.append(float(condition.WEIGHT))

        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('PHASE_MIX weights must be finite, nonnegative, and nonzero')
        self.phase_probabilities = weights / weights.sum()
        if self.logger is not None:
            summary = ', '.join(
                f'{root.name}:{probability:.4f}'
                for root, probability in zip(self.phase_roots, self.phase_probabilities)
            )
            self.logger.info('Boreas PR-DSVT phase mix: %s', summary)

    def get_lidar(self, idx):
        if not self.phase_mix_enabled:
            return super().get_lidar(idx)
        selected = np.random.choice(
            len(self.phase_roots), p=self.phase_probabilities
        )
        lidar_file = self.phase_roots[selected] / 'points' / f'{idx}.npy'
        if not lidar_file.is_file():
            raise FileNotFoundError(lidar_file)
        return np.load(lidar_file)
