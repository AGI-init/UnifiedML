# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Agents.Blocks.Architectures.Transformer import SelfAttentionBlock, LearnableFourierPositionalEncodings
from Agents.Blocks.Architectures.Vision.CNN import AvgPool, CNN


class ViT(nn.Module):
    """
    A Vision Transformer (https://arxiv.org/abs/2010.11929)
    Generalized to adapt to arbitrary temporal-spatial dimensions, assumes channels-first
    """
    def __init__(self, input_shape=(32, 7, 7), out_channels=32, patch_size=4, num_heads=None, depth=3, emb_dropout=0.1,
                 query_key_dim=None, mlp_hidden_dim=None, dropout=0.1, pool_type='cls', output_shape=None, fourier=False):
        super().__init__()

        output_dim = Utils.prod(output_shape)

        self.num_axes = len(input_shape)

        # Convolve into patches - image/input spatial dims should ideally be dividable by patch size(s)
        self.Vi = CNN(input_shape, out_channels, 0, last_relu=False, kernel_size=patch_size, stride=patch_size)
        shape = Utils.cnn_feature_shape(input_shape, self.Vi)

        positional_encodings = (LearnableFourierPositionalEncodings if fourier
                                else LearnablePositionalEncodings)(shape)

        token = CLSToken(shape)  # Just appends a parameterized token
        shape = Utils.repr_shape(shape, token)

        # Positional encoding -> CLS Token -> Attention layers
        self.T = nn.Sequential(positional_encodings,
                               token,
                               nn.Dropout(emb_dropout),

                               # Transformer
                               *[SelfAttentionBlock(shape, num_heads, None, query_key_dim, mlp_hidden_dim, dropout)
                                 for _ in range(depth)],

                               # Can CLS-pool and project to a specified output dim, optional
                               nn.Identity() if output_dim is None else nn.Sequential(CLSPool() if pool_type == 'cls'
                                                                                      else AvgPool(),
                                                                                      nn.Linear(out_channels,
                                                                                                output_dim)))

    def shape(self, shape):
        return Utils.cnn_feature_shape(shape, self.Vi, self.T)

    def forward(self, *x):
        patches = self.Vi(*x)

        # Conserve leading dims, operate on batch-item dims
        lead_dims = patches.shape[:-self.num_axes]

        outs = self.T(patches.flatten(0, -self.num_axes - 1))

        # Restore lead shape
        return outs.view(*lead_dims, *outs.shape[1:])


class LearnablePositionalEncodings(nn.Module):
    def __init__(self, input_shape):
        """Learnable positional encodings. Generalized to adapt to arbitrary dimensions. Assumes channels-first!"""
        super().__init__()

        self.encoding = nn.Parameter(torch.randn(*input_shape))

    def shape(self, shape):
        return shape  # Conserves shape

    def forward(self, x):
        return x + self.encoding


class CLSToken(nn.Module):
    """Appends a CLS token, assumes channels-first (https://arxiv.org/pdf/1810.04805.pdf)"""
    def __init__(self, input_shape=(32,)):
        super().__init__()

        in_channels, *self.spatial_dims = input_shape

        self.token = nn.Parameter(torch.randn(in_channels, 1))

    def shape(self, shape):
        c, *shape = shape
        return c, math.prod(shape) + 1

    def forward(self, obs):
        return torch.cat([obs.flatten(-len(self.spatial_dims)), self.token.expand(*obs.shape[:-len(self.spatial_dims)], 1)], dim=-1)  # Assumes 2 spatial dims


class CLSPool(nn.Module):
    """Selects (indexes) the CLS token as the representative embedding, assuming channels-first"""
    def __init__(self, **_):
        super().__init__()

    def shape(self, shape):
        c = shape[0]
        return c,

    def forward(self, x):
        return x.flatten(2)[..., -1]

