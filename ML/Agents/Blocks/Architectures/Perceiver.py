# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch

from torch import nn

import Utils

from Agents.Blocks.Architectures import MLP
from Agents.Blocks.Architectures.Transformer import AttentionBlock, LearnableFourierPositionalEncodings, PositionalEncodings


class Perceiver(nn.Module):
    """Perceiver (https://arxiv.org/abs/2103.03206) (https://arxiv.org/abs/2107.14795)
    Generalized to arbitrary spatial dimensions, dimensionality-agnostic I/O w.r.t. state dict.
    For consistency with Vision models, assumes channels-first!"""
    def __init__(self, input_shape=(64,), num_tokens=64, num_heads=None, token_dim=None, output_shape=None,
                 depths=None, recursions=None, learnable_tokens=True, channels_first=True,
                 learnable_positional_encodings=False, positional_encodings=True):
        super().__init__()

        # Adapt to proprioceptive
        input_shape = (1,) * channels_first + ((input_shape,) if isinstance(input_shape, int)
                                               else tuple(input_shape)) + (1,) * (not channels_first) \
            if len(input_shape) < 2 else input_shape

        # Adapt to variable-len input

        positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
            else PositionalEncodings if positional_encodings else nn.Identity

        self.positional_encodings = positional_encodings(input_shape, channels_first=channels_first)

        # Dimensions

        self.channels_first = channels_first

        self.num_tokens = num_tokens
        self.output_dim = Utils.prod(output_shape)

        shape = Utils.repr_shape(input_shape, self.positional_encodings)

        self.input_dim = shape if isinstance(shape, int) else shape[0] if channels_first else shape[-1]
        self.token_dim = token_dim or self.input_dim

        depths = [depths] if isinstance(depths, int) else depths if depths else [3]
        recursions = [recursions] if isinstance(recursions, int) else recursions if recursions else [1] * len(depths)

        assert len(depths) == len(recursions), f'Recursion must be specified for each depth: {recursions}, {depths}'
        assert self.token_dim == self.input_dim or recursions[0] == 1, \
            f'First depth cannot be recursive if token_dim ≠ input_dim, {self.token_dim}≠{self.input_dim}'

        # Input tokens

        tokens = torch.zeros(1, self.token_dim, self.num_tokens) if self.channels_first \
            else torch.zeros(1, self.num_tokens, self.token_dim)

        if learnable_tokens:
            self.tokens = nn.Parameter(tokens)
        else:
            tokens = PositionalEncodings(tokens.shape[1:], 0, channels_first=channels_first)(tokens)
            self.register_buffer('tokens', tokens, persistent=False)

        if learnable_tokens:
            nn.init.kaiming_uniform_(self.tokens, a=math.sqrt(5))

        # Perceiver attention layers

        self.cross_attention = nn.ModuleList(sum([[AttentionBlock(self.token_dim, num_heads, self.input_dim,
                                                                  channels_first=channels_first)] * recurs
                                                  for i, recurs in enumerate(recursions)], []))

        self.self_attentions = nn.ModuleList(sum([[nn.Sequential(*[AttentionBlock(self.token_dim, num_heads,
                                                                                  channels_first=channels_first)
                                                                   for _ in range(depth - 1)])] * recurs
                                                  for recurs, depth in zip(recursions, depths)], []))

        # Output tokens

        if self.output_dim is not None:
            outputs = torch.randn(1, self.output_dim, self.token_dim)  # Max output dim
            self.register_buffer('outputs', outputs, persistent=False)

            self.to_outputs = LearnableFourierPositionalEncodings(self.outputs.shape[1:], channels_first=False)

            self.output_attention = AttentionBlock(self.token_dim, num_heads, self.token_dim, channels_first=False)

            self.MLP = MLP(self.token_dim, 1, self.token_dim, 1, activation=nn.GELU())

    def shape(self, shape):
        # Passed-in output dim, or same shape as tokens
        return (self.output_dim,) if self.output_dim else (self.token_dim, self.num_tokens) if self.channels_first \
            else (self.num_tokens, self.token_dim)

    def forward(self, input, output_dim=None):
        # Adapt proprioceptive
        if len(input.shape) < 3:
            input = input.unsqueeze(1 if self.channels_first else -1)

        input = self.positional_encodings(input)
        output = self.tokens

        for cross_attention, self_attentions in zip(self.cross_attention, self.self_attentions):
            output = self_attentions(cross_attention(output, input))

        # Dimensionality-adaptive output

        if self.output_dim:
            if self.channels_first:
                output = Utils.ChSwap(output)

            output = self.MLP(self.output_attention(self.to_outputs(self.outputs), output)).squeeze(-1)

        return output
