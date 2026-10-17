"""Phase-robust extensions for the DSVT backbone."""

import math

import torch
import torch.nn as nn

from .dsvt import DSVT


class PhaseRobustDSVT(DSVT):
    """DSVT with angular voxel encoding and observed beam-profile conditioning.

    The conditioner observes only XYZ-derived elevation statistics. It does not
    require a sensor name or a ring-id feature at inference time.
    """

    def __init__(self, model_cfg, input_channels, voxel_size, point_cloud_range,
                 **kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, **kwargs)
        if input_channels != model_cfg.d_model[0]:
            raise ValueError(
                f'PR-DSVT input width {input_channels} must equal d_model[0] '
                f'{model_cfg.d_model[0]}'
            )

        profile_cfg = model_cfg.PROFILE_CONDITIONING
        self.profile_bins = int(profile_cfg.get('NUM_BINS', 64))
        elevation_range = profile_cfg.get('ELEVATION_RANGE_DEG', [-30.0, 20.0])
        self.elevation_min = math.radians(float(elevation_range[0]))
        self.elevation_max = math.radians(float(elevation_range[1]))
        hidden = int(profile_cfg.get('HIDDEN_CHANNELS', 64))
        strength = float(profile_cfg.get('MAX_MODULATION', 0.25))
        self.max_modulation = strength

        self.profile_mlp = nn.Sequential(
            nn.Linear(self.profile_bins, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_channels * 2),
        )
        self.angular_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_channels),
        )
        local_cfg = model_cfg.get('LOCAL_RELIABILITY', {})
        self.use_local_reliability = bool(local_cfg.get('ENABLED', False))
        if self.use_reliability_attention and not self.use_local_reliability:
            raise ValueError(
                'RELIABILITY_ATTENTION requires LOCAL_RELIABILITY.ENABLED'
            )
        if self.use_local_reliability:
            self.local_support_bins = int(local_cfg.get('NUM_BINS', 8))
            local_hidden = int(local_cfg.get('HIDDEN_CHANNELS', 64))
            self.local_max_modulation = float(
                local_cfg.get('MAX_MODULATION', 0.20)
            )
            self.local_count_normalizer = float(
                local_cfg.get('COUNT_NORMALIZER', 128.0)
            )
            self.local_mlp = nn.Sequential(
                nn.Linear(self.local_support_bins + 7, local_hidden),
                nn.ReLU(),
                nn.Linear(local_hidden, input_channels * 2 + 1),
            )

        # Start exactly from the unconditioned DSVT behavior.
        nn.init.zeros_(self.profile_mlp[-1].weight)
        nn.init.zeros_(self.profile_mlp[-1].bias)
        nn.init.zeros_(self.angular_mlp[-1].weight)
        nn.init.zeros_(self.angular_mlp[-1].bias)
        if self.use_local_reliability:
            nn.init.zeros_(self.local_mlp[-1].weight)
            nn.init.zeros_(self.local_mlp[-1].bias)

        self.register_buffer('voxel_size_xyz', torch.as_tensor(voxel_size, dtype=torch.float32))
        self.register_buffer(
            'point_cloud_range_xyz',
            torch.as_tensor(point_cloud_range, dtype=torch.float32),
        )

    def _beam_profile(self, points, batch_size, dtype):
        xyz = points[:, 1:4]
        horizontal = torch.linalg.vector_norm(xyz[:, :2], dim=1).clamp_min(1e-3)
        elevation = torch.atan2(xyz[:, 2], horizontal)
        scaled = (elevation - self.elevation_min) / (
            self.elevation_max - self.elevation_min
        )
        bins = torch.floor(scaled * self.profile_bins).long()
        valid = (bins >= 0) & (bins < self.profile_bins)
        batch_index = points[:, 0].long()[valid]
        flat_index = batch_index * self.profile_bins + bins[valid]
        histogram = points.new_zeros(batch_size * self.profile_bins)
        histogram.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=histogram.dtype))
        histogram = histogram.view(batch_size, self.profile_bins)
        histogram = torch.log1p(histogram)
        histogram = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return histogram.to(dtype=dtype)

    def _voxel_angular_features(self, coords, dtype):
        # coords are [batch, z, y, x]. Use voxel centers in metric space.
        xyz_index = coords[:, [3, 2, 1]].to(dtype=dtype)
        xyz = self.point_cloud_range_xyz[:3].to(dtype=dtype) + (
            xyz_index + 0.5
        ) * self.voxel_size_xyz.to(dtype=dtype)
        horizontal = torch.linalg.vector_norm(xyz[:, :2], dim=1).clamp_min(1e-3)
        distance = torch.linalg.vector_norm(xyz, dim=1).clamp_min(1e-3)
        elevation = torch.atan2(xyz[:, 2], horizontal)
        range_scale = torch.linalg.vector_norm(
            self.point_cloud_range_xyz[3:5] - self.point_cloud_range_xyz[:2]
        ).to(dtype=dtype).clamp_min(1.0)
        return torch.stack((
            torch.sin(elevation),
            torch.cos(elevation),
            torch.log1p(distance) / torch.log1p(range_scale),
            xyz[:, 2] / max(
                abs(float(self.point_cloud_range_xyz[2])),
                abs(float(self.point_cloud_range_xyz[5])),
                1.0,
            ),
        ), dim=1)

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = int(batch_dict['batch_size'])
        profile = self._beam_profile(
            batch_dict['points'], batch_size, voxel_features.dtype
        )
        modulation = self.profile_mlp(profile)
        gamma, beta = modulation.chunk(2, dim=1)
        gamma = self.max_modulation * torch.tanh(gamma)
        beta = self.max_modulation * torch.tanh(beta)
        voxel_batch = voxel_coords[:, 0].long()

        angular = self._voxel_angular_features(voxel_coords, voxel_features.dtype)
        angular_embedding = self.angular_mlp(angular)
        conditioned = (
            voxel_features * (1.0 + gamma[voxel_batch])
            + beta[voxel_batch]
            + angular_embedding
        )

        if self.use_local_reliability:
            required = (
                'voxel_angular_histogram', 'voxel_point_count',
                'voxel_support_coverage', 'voxel_support_entropy',
            )
            missing = [name for name in required if name not in batch_dict]
            if missing:
                raise KeyError(
                    f'LOCAL_RELIABILITY requires VFE descriptors: {missing}'
                )

            support_histogram = batch_dict['voxel_angular_histogram'].to(
                dtype=voxel_features.dtype
            )
            if support_histogram.shape[1] != self.local_support_bins:
                raise ValueError(
                    'LOCAL_RELIABILITY.NUM_BINS must match '
                    'VFE.SUPPORT_ENCODING.NUM_BINS'
                )
            density = torch.log1p(
                batch_dict['voxel_point_count'].to(voxel_features.dtype)
            ) / math.log1p(self.local_count_normalizer)
            density = density.clamp(0.0, 1.0)
            coverage = batch_dict['voxel_support_coverage'].to(
                voxel_features.dtype
            )
            entropy = batch_dict['voxel_support_entropy'].to(
                voxel_features.dtype
            )
            local_input = torch.cat((
                support_histogram,
                density[:, None],
                coverage[:, None],
                entropy[:, None],
                angular,
            ), dim=1)
            local_modulation = self.local_mlp(local_input)
            local_gamma, local_beta, reliability_correction = torch.split(
                local_modulation,
                [voxel_features.shape[1], voxel_features.shape[1], 1],
                dim=1,
            )
            local_gamma = self.local_max_modulation * torch.tanh(local_gamma)
            local_beta = self.local_max_modulation * torch.tanh(local_beta)
            conditioned = conditioned * (1.0 + local_gamma) + local_beta

            base_reliability = (
                0.40 * density + 0.30 * coverage + 0.30 * entropy
            ).clamp(0.05, 0.95)
            base_logit = torch.log(base_reliability) - torch.log1p(
                -base_reliability
            )
            reliability = torch.sigmoid(
                base_logit + reliability_correction[:, 0]
            )
            batch_dict['voxel_reliability'] = reliability

        batch_dict['voxel_features'] = conditioned
        batch_dict['beam_profile'] = profile.detach()
        return super().forward(batch_dict)
