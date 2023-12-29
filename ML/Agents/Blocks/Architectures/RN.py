# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

import Utils

from Agents.Blocks.Architectures.MLP import MLP
from Agents.Blocks.Architectures.Transformer import PositionalEncodings, LearnableFourierPositionalEncodings


class RN(nn.Module):
    """Relation Network (https://arxiv.org/abs/1706.01427)
    Adapts to arbitrary spatial dims. Un-pooled by default except over contexts (no "outer" MLP), outputs a feature map.
    Supports positional encodings. For consistency with Vision models, assumes channels-first!"""
    def __init__(self, input_shape=(32,), context_dim=None, depth=1, hidden_dim=None, output_shape=None, dropout=0,
                 channels_first=True, learnable_positional_encodings=False, positional_encodings=True):
        super().__init__()

        positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
            else PositionalEncodings if positional_encodings else nn.Identity

        self.positional_encodings = positional_encodings(input_shape, channels_first=channels_first)

        # Dimensions

        self.channels_first = channels_first

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        shape = Utils.repr_shape(input_shape, self.positional_encodings)

        self.input_dim = shape[0] if channels_first else shape[-1]

        # Defaults
        self.context_dim = context_dim or self.input_dim
        self.hidden_dim = hidden_dim or self.input_dim * 4
        self.output_dim = Utils.prod(output_shape) or self.input_dim

        self.inner = MLP(self.input_dim + self.context_dim, self.output_dim, self.hidden_dim, depth)

        self.dropout = nn.Dropout(dropout)

    def shape(self, shape):  # Conserves spatial dimensions, maps channel dim to output-dim
        return (self.output_dim, *shape[1:]) if self.channels_first else (*shape[:-1], self.output_dim)

    def forward(self, input, context=None):
        input = self.positional_encodings(input)

        if context is None:
            context = input  # Self-relation

        # Permute as channels-last
        if self.channels_first:
            input, context = [Utils.ChSwap(x, False) for x in [input, context]]

        assert input.shape[-1] == self.input_dim, f'Unexpected input shape {input.shape[-1]}≠{self.input_dim}'
        assert context.shape[-1] == self.context_dim, f'Unexpected context shape {context.shape[-1]}≠{self.context_dim}'

        # Preserve batch/spatial dims
        lead_dims = input.shape[:-1]

        # Flatten intermediary spatial dims, combine input & context pairwise
        pairs = Utils.batched_cartesian_prod([input.flatten(1, -2), context.flatten(1, -2)],
                                             dim=1, collapse_dims=False).flatten(-2)

        relations = self.dropout(self.inner(pairs)).sum(1)  # Pool over contexts

        # Restores original leading dims
        output = relations.view(*lead_dims, -1)

        # Convert to channels-first
        if self.channels_first:
            output = Utils.ChSwap(output, False)

        return output
