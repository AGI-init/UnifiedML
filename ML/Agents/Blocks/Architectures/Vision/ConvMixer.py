# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch.nn as nn

from Agents.Blocks.Architectures.Residual import Residual
from Agents.Blocks.Architectures.Vision.CNN import AvgPool, cnn_broadcast

import Utils


class ConvMixer(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, kernel_size=9, patch_size=7, output_shape=None):
        super().__init__()

        self.input_shape, output_dim = Utils.to_tuple(input_shape), Utils.prod(output_shape)

        in_channels = self.input_shape[0]

        self.ConvMixer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size, groups=out_channels, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(out_channels)
                )),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(depth)])

        self.repr = nn.Identity() if output_dim is None \
            else nn.Sequential(AvgPool(), nn.Linear(out_channels, output_dim))  # Optional output projection

    def shape(self, shape):
        return Utils.cnn_feature_shape(shape, self.ConvMixer, self.repr)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.ConvMixer(x)
        x = self.repr(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
