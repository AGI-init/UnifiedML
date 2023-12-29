# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

from Agents.Blocks.Architectures.Residual import Residual
from Agents.Blocks.Architectures.Vision.CNN import AvgPool, cnn_broadcast

import Utils


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, down_sample=None):
        super().__init__()

        if down_sample is None and (in_channels != out_channels or stride != 1):
            down_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(out_channels))

        block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=kernel_size // 2,
                                        stride=stride, bias=False),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(out_channels, out_channels,
                                        kernel_size=kernel_size, padding=kernel_size // 2,
                                        bias=False),
                              nn.BatchNorm2d(out_channels))

        self.ResBlock = nn.Sequential(Residual(block, down_sample),
                                      nn.ReLU(inplace=True))

    def shape(self, shape):
        return Utils.cnn_feature_shape(shape, self.ResBlock)

    def forward(self, x):
        return self.ResBlock(x)


class MiniResNet(nn.Module):
    """
    A ResNet backbone with computationally-efficient defaults.
    """
    def __init__(self, input_shape, kernel_size=3, stride=2, dims=(32, 32), depths=(3,), output_shape=None):
        super().__init__()

        self.input_shape, output_dim = Utils.to_tuple(input_shape), Utils.prod(output_shape)

        in_channels = self.input_shape[0]

        if kernel_size % 2 == 0:
            kernel_size += 1  # Odd kernel value

        # ResNet
        self.ResNet = nn.Sequential(nn.Conv2d(in_channels, dims[0],
                                              kernel_size=kernel_size, padding=1, bias=False),
                                    # kernel_size=7, stride=2, padding=3, bias=False),  # Common settings
                                    nn.BatchNorm2d(dims[0]),
                                    nn.ReLU(inplace=True),
                                    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Common settings
                                    *[nn.Sequential(*[ResBlock(dims[i + (j > 0)], dims[i + 1], kernel_size,
                                                               1 + (stride - 1) * (i > 0 and j > 0))
                                                      for j in range(depth)])
                                      for i, depth in enumerate(depths)])

        self.repr = nn.Identity() if output_dim is None \
            else nn.Sequential(AvgPool(), nn.Linear(dims[-1], output_dim))  # Project to desired shape

    def shape(self, shape):
        return Utils.cnn_feature_shape(shape, self.ResNet, self.repr)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.ResNet(x)
        x = self.repr(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class ResNet18(MiniResNet):
    def __init__(self, input_shape, output_shape=None):
        super().__init__(input_shape, 3, 2, [64, 64, 128, 256, 512], [2, 2, 2, 2], output_shape)


class ResNet50(MiniResNet):
    def __init__(self, input_shape, output_shape=None):
        super().__init__(input_shape, 3, 2, [64, 64, 128, 256, 512], [3, 4, 6, 3], output_shape)
