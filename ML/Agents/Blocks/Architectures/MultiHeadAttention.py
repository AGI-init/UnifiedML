# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

from einops import rearrange

import torch
from torch import nn

import Utils


class Attention(nn.Module):
    """
    Multi-head dot-product attention (MHDPA) from inputs to contexts (Cross-Attention)
    (https://arxiv.org/abs/1706.03762?context=cs)

    All you need

    Generalized to any-dimensionality input shapes, and includes options for "talking heads" and "ReLA".
    For consistency with Vision models, defaults to channels-first! Assumes input & context have batch + spatial dim(s).
    """
    def __init__(self, input_shape=(32,), num_heads=None, context_dim=None, query_key_dim=None, value_dim=None,
                 talking_heads=False, rela=False, channels_first=True):
        super().__init__()

        self.channels_first = channels_first

        # Dimensions
        self.input_dim = input_shape if isinstance(input_shape, int) \
            else input_shape[0] if channels_first else input_shape[-1]

        # Defaults
        self.context_dim = context_dim or self.input_dim
        self.query_key_dim = query_key_dim or self.input_dim
        self.value_dim = value_dim or self.input_dim

        self.num_heads = num_heads or math.gcd(math.gcd(16, self.value_dim), self.query_key_dim)

        assert self.value_dim % self.num_heads == self.query_key_dim % self.num_heads == 0, \
            f'Value dim={self.value_dim}, QueryKey dim={self.query_key_dim} must be divisible by heads={self.num_heads}'

        # Linear QKV-projections (Perhaps just substitute with the more general Conv)
        self.to_query = nn.Linear(self.input_dim, self.query_key_dim, bias=False)
        self.to_key_value = nn.Linear(self.context_dim, self.query_key_dim + self.value_dim, bias=False)

        # Can access attention weights
        self.saved_attention_weights = None

        # Additional options

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        if talking_heads:
            self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(self.num_heads, self.num_heads, bias=False),
                                        nn.LayerNorm(self.num_heads), Utils.ChSwap)

        # "Rectified-Linear Attention (ReLA)" (https://arxiv.org/abs/2104.07012)
        if rela:
            self.rela = nn.ReLU(inplace=True)

    def shape(self, shape):
        # Conserves spatial dimensions, maps channel dim to value-dim
        return (self.value_dim, *shape[1:]) if self.channels_first \
            else (*shape[:-1], self.value_dim)

    def forward(self, input, context=None):
        if context is None:
            context = input  # Self-attention

        # Permute as channels-last
        if self.channels_first:
            input, context = [Utils.ChSwap(x, False) for x in [input, context]]

        # Preserve spatial dims
        spatial_dims = input.shape[1:-1]

        # Flatten intermediary spatial dims
        input = input.flatten(1, -2)
        context = context.flatten(1, -2)

        # Validate shapes
        assert input.shape[-1] == self.input_dim, f'Unexpected input shape {input.shape[-1]}≠{self.input_dim}'
        assert context.shape[-1] == self.context_dim, f'Unexpected context shape {context.shape[-1]}≠{self.context_dim}'

        query = self.to_query(input)
        key, value = self.to_key_value(context).tensor_split((self.query_key_dim,), -1)  # Split into KV

        # Heads-first
        query, key, value \
            = [rearrange(qkv, 'b n (h d) -> b h n d', h=self.num_heads) for qkv in (query, key, value)]

        # Scale (Q / sqrt(d))
        query *= query.shape[-1] ** -0.5

        # Multiply (W = Q * K)
        self.saved_attention_weights = torch.einsum('b h i d, b h j d -> b h i j', query, key)

        # Normalize - disabled for now
        # if not hasattr(self, 'rela'):
        #     self.saved_attention_weights -= self.saved_attention_weights.amax(-1, keepdim=True).detach()

        # Softmax
        attention_weights = self.rela(self.saved_attention_weights) if hasattr(self, 'rela') \
            else self.saved_attention_weights.softmax(dim=-1)

        # "Talking heads"
        if hasattr(self, 'talk_h'):
            attention_weights = self.talk_h(attention_weights)

        # Attend (W * V)
        attention = torch.matmul(attention_weights, value)

        # Heads-last-concatenated
        output = rearrange(attention, 'b h n d -> b n (h d)')

        # Restores original leading dims
        output = output.view(output.shape[0], *spatial_dims, -1)

        # Convert to channels-first
        if self.channels_first:
            output = Utils.ChSwap(output, False)

        return output


class MHDPA(Attention):
    """Pseudonym"""


class CrossAttention(Attention):
    """Cross-attention, pseudonym for Attention"""


class SelfAttention(Attention):
    """Self-attention, just cross-attention except context = input"""
    def forward(self, input, *_):
        return super().forward(input)


class ReLA(Attention):
    """ReLA: Rectified linear attention (https://arxiv.org/abs/2104.07012)"""
    def __init__(self, input_shape=(32,), num_heads=None, context_dim=None, query_key_dim=None, value_dim=None,
                 talking_heads=False, channels_first=True):
        super().__init__(input_shape, num_heads, context_dim, query_key_dim, value_dim, talking_heads,
                         True, channels_first)
