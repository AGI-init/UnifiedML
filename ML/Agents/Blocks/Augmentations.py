# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, obs):
        # Operates on last 3 dims of x, preserves leading dims
        shape = obs.shape
        assert len(shape) > 3, f'Obs shape {tuple(shape)} not supported by this augmentation, try \'aug=Identity\''
        obs = obs.view(-1, *shape[-3:])
        n, c, h, w = obs.size()
        assert h == w, f'Height≠width ({h}≠{w}), obs shape not supported by this augmentation, try \'aug=Identity\''
        padding = tuple([self.pad] * 4)
        if not torch.is_floating_point(obs):
            obs = obs.to(torch.float32)  # Cuda replication_pad2d_cuda requires float
        obs = F.pad(obs, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=obs.device,
                                dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=obs.device,
                              dtype=obs.dtype).float()
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift

        device = obs.device.type

        warnings.filterwarnings("ignore", message='.*grid_sampler_2d op is supported natively starting from macOS 13.1')

        output = F.grid_sample(obs.float() if obs.device.type == 'cpu' else obs,
                               grid,
                               padding_mode='zeros',
                               align_corners=False).to(device)

        obs = output.view(*shape[:-3], *output.shape[-3:])

        return obs


class IntensityAug(nn.Module):
    def __init__(self, scale=0.1, noise=2):
        super().__init__()
        self.scale, self.noise = scale, noise

    def forward(self, obs):
        axes = (1,) * len(obs.shape[2:])  # Spatial axes, useful for dynamic input shapes
        noise = 1.0 + (self.scale * torch.randn(
            (obs.shape[0], 1, *axes), device=obs.device).clamp_(-self.noise, self.noise))  # Random noise
        obs = obs * noise

        return obs
