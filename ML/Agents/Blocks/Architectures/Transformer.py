# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Agents.Blocks.Architectures.MLP import MLP
from Agents.Blocks.Architectures.MultiHeadAttention import CrossAttention


class AttentionBlock(nn.Module):
    """
    A Transformer pre-norm block (https://arxiv.org/pdf/2002.04745.pdf)
    Generalized to cross-attend from inputs to contexts, broadcasting various shapes, with support for "talking heads"
    and "ReLA". For consistency with Vision models, assumes channels-first!
    """
    def __init__(self, input_shape=(32,), num_heads=None, context_dim=None, query_key_dim=None, mlp_hidden_dim=None,
                 dropout=0, talking_heads=False, rela=False, channels_first=True):
        super().__init__()

        self.channels_first = channels_first

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        if channels_first:
            input_shape = list(reversed(input_shape))  # Move channel dim to last

        # Multi-Head Dot-Product Attention (MHDPA) from inputs to context
        self.attend = CrossAttention(input_shape, num_heads, context_dim, query_key_dim, None, talking_heads, rela,
                                     channels_first=False)

        self.LayerNormPre = nn.LayerNorm(self.attend.input_dim)  # Applied before the above attention

        # "Rectified-Linear Attention (ReLA)" (https://arxiv.org/abs/2104.07012)
        if rela:
            self.LayerNormReLA = nn.LayerNorm(self.attend.value_dim)

        if self.attend.num_heads > 1:
            self.map_heads = nn.Sequential(nn.Linear(self.attend.value_dim, self.attend.input_dim),
                                           nn.Dropout(dropout))

        self.LayerNormPost = nn.LayerNorm(self.attend.input_dim)

        self.mlp_hidden_dim = mlp_hidden_dim or self.attend.value_dim * 4  # MLP dimension

        self.MLP = nn.Sequential(MLP(self.attend.input_dim, self.attend.input_dim, self.mlp_hidden_dim,
                                     depth=1, activation=nn.GELU(), dropout=dropout), nn.Dropout(dropout))

    def shape(self, shape):
        # Isotropic, conserves dimensions
        return shape

    def forward(self, input, context=None):
        # To channels-last
        if self.channels_first:
            input = Utils.ChSwap(input, False)

            if context is not None:
                context = Utils.ChSwap(context, False)

        pre_norm = self.LayerNormPre(input)

        if context is None:
            context = pre_norm

        attention = self.attend(pre_norm, context)

        if hasattr(self, 'LayerNormReLA'):
            attention = self.LayerNormReLA(attention)

        if hasattr(self, 'map_heads'):
            attention = self.map_heads(attention)

        residual = attention + input
        output = self.MLP(self.LayerNormPost(residual)) + residual

        return Utils.ChSwap(output, False) if self.channels_first \
            else output


class CrossAttentionBlock(AttentionBlock):
    """Cross-Attention Block, same as the Attention Block"""


class SelfAttentionBlock(AttentionBlock):
    """A.K.A. a Transformer pre-norm block, same as the Cross-Attention Block except input = context"""
    def forward(self, input, *_):
        return super().forward(input)


class LearnableFourierPositionalEncodings(nn.Module):
    def __init__(self, input_shape=(32,), fourier_dim=None, hidden_dim=None, output_shape=None, channels_first=True):
        """
        Learnable Fourier Features (https://arxiv.org/pdf/2106.02795.pdf)
        Generalized to adapt to arbitrary spatial dimensions. For consistency with Vision models,
        assumes channels-first!
        """
        super().__init__()

        # Dimensions

        self.channels_first = channels_first

        self.input_dim = input_shape if isinstance(input_shape, int) \
            else input_shape[0] if channels_first else input_shape[-1]

        fourier_dims = fourier_dim or self.input_dim
        self.fourier_dim = -(-fourier_dims // 2)  # Round up

        self.hidden_dim = hidden_dim or self.input_dim
        self.output_dim = Utils.prod(output_shape) or self.input_dim

        self.scale = 1 / math.sqrt(self.fourier_dim)

        # Projections
        self.Linear = nn.Linear(self.input_dim, self.fourier_dim, bias=False)
        self.MLP = MLP(self.fourier_dim * 2, self.output_dim, self.hidden_dim, 1, nn.GELU())

        # Initialize weights
        nn.init.normal_(self.Linear.weight.data)

    def shape(self, shape):
        # Conserves spatial dimensions
        return (self.output_dim, *shape[1:]) if self.channels_first \
            else (*shape[:-1], self.output_dim)

    def forward(self, input):
        # Permute as channels-last
        if self.channels_first:
            input = Utils.ChSwap(input, False)

        # Linear-project features
        features = self.Linear(input)

        cosines, sines = torch.cos(features), torch.sin(features)
        cosines *= self.scale
        sines *= self.scale

        # Fourier features via MLP
        fourier_features = self.MLP(cosines, sines)

        # To channels-first if needed
        output = Utils.ChSwap(fourier_features, False) if self.channels_first \
            else fourier_features

        return output


class PositionalEncodings(nn.Module):
    """
    Sine-cosine positional encodings
    Generalized to adapt to arbitrary spatial dimensions. For consistency with Vision models,
    assumes channels-first! Automatically additive when encoding size is same as input, otherwise concatenates.
    """
    def __init__(self, input_shape=(7, 32, 32), dropout=0.1, size=None, max_spatial_lens=None, channels_first=True):
        super().__init__()

        self.channels_first = channels_first

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        # Max spatial lengths (for variable-len sequences)
        if max_spatial_lens is None:
            max_spatial_lens = input_shape[-1:None:-1][:-1] if channels_first else input_shape[:-1]

        # Dimensions
        self.input_dim = input_shape[0] if channels_first else input_shape[-1]
        self.size = max(size or self.input_dim, len(max_spatial_lens) * 2)

        div_term = torch.exp(torch.arange(0, self.size, len(max_spatial_lens) * 2) * (-math.log(10000.0) / self.size))

        positions = torch.stack(torch.meshgrid(*map(torch.arange, max_spatial_lens), indexing='ij'), -1)
        positions = positions.unsqueeze(-1).float().matmul(div_term.unsqueeze(-2)).flatten(-2)

        positional_encodings = torch.zeros(*max_spatial_lens, self.size)
        positional_encodings[..., 0::2] = torch.sin(positions)
        positional_encodings[..., 1::2] = torch.cos(positions[..., :-(self.size % 2) or None])  # Odds

        self.register_buffer('positional_encodings', positional_encodings, persistent=False)

        self.dropout = nn.Dropout(p=dropout)

    def shape(self, shape):
        # Conserves shape when additive (if spatial axes * 2 ≤ input dim), else concatenates
        return shape if self.input_dim == self.size \
            else (self.input_dim + self.size, *shape[1:]) if self.channels_first \
            else (*shape[:-1], self.input_dim + self.size)

    def forward(self, input):
        # Permute as channels-last
        if self.channels_first:
            input = Utils.ChSwap(input, False)

        positions = self.positional_encodings[list(map(slice, input.shape[1:-1]))]

        # Add or concatenate
        encodings = self.dropout(input + positions) if self.input_dim == self.size \
            else torch.cat([input, positions.expand(input.shape[0], *positions.shape)], -1)

        # To channels-first if needed
        return Utils.ChSwap(encodings, False) if self.channels_first \
            else encodings


class Transformer(nn.Module):
    """A Transformer
    For consistency with Vision models, assumes channels-first!
    Generalized to arbitrary spatial dimensions"""
    def __init__(self, input_shape=(32,), num_heads=None, depth=1, channels_first=True,
                 learnable_positional_encodings=False, positional_encodings=True):
        super().__init__()

        positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
            else PositionalEncodings if positional_encodings else nn.Identity

        positional_encodings = positional_encodings(input_shape, channels_first=channels_first)

        self.shape = Utils.repr_shape(input_shape, positional_encodings)

        self.transformer = nn.Sequential(positional_encodings, *[SelfAttentionBlock(self.shape, num_heads,
                                                                                    channels_first=channels_first)
                                                                 for _ in range(depth)])

    def shape(self, shape):
        return self.shape

    def forward(self, obs):
        return self.transformer(obs)
