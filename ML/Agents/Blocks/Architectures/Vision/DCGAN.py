# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

import Utils

from Agents.Blocks.Architectures.Vision.CNN import cnn_broadcast


"""
Usage example: 
python Run.py task=classify/mnist generate=true Discriminator=DCGAN.Discriminator Generator=DCGAN.Generator

Note: Dimensionality adaptivity to input shapes is still highly experimental for GANs/DCGAN.
"""


class Generator(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)
        # Proprioceptive is channel dim
        self.input_shape = tuple(self.input_shape) + (1,) * (3 - len(self.input_shape))  # Broadcast input to 2D

        in_channels = self.input_shape[0]
        out_channels = in_channels if self.output_shape is None else self.output_shape[0]

        self.Generator = nn.Sequential(
            # (hidden_dim * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels, hidden_dim * 8, 4, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),

            # (hidden_dim * 4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),

            # (hidden_dim * 2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            # hidden_dim x 32 x 32
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # out_channels x 64 x 64
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
            nn.Identity() if self.output_shape is None else nn.AdaptiveAvgPool2d(self.output_shape[1:])  # Adapts scale
        )

        self.apply(weight_init)

    def shape(self, shape):
        c = shape[0]
        return c, 64, 64  # cnn_feature_shape doesn't support pre-computing ConvTranspose2d

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.Generator(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)

        in_channels = self.input_shape[0]

        self.Discriminator = nn.Sequential(
            # hidden_dim x 32 x 32
            nn.AdaptiveAvgPool2d(64),  # Adapts from different scales
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 2) x 16 x 16
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 4) x 8 x 8
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 8) x 4 x 4
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 1 x 1 x 1
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(weight_init)

    def shape(self, shape):
        return Utils.cnn_feature_shape(shape, self.Discriminator)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.Discriminator(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out


# Initializes model weights a la normal
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
