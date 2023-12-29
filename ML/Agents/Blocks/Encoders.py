# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy

import torch
from torch import nn

from minihydra import instantiate

from Agents.Blocks.Architectures.Vision.CNN import CNN

import Utils


class CNNEncoder(nn.Module):
    """
    CNN encoder generalized to work with proprioceptive/spatial inputs and multi-dimensionality convolutions (1d or 2d)
    """
    def __init__(self, obs_spec, context_dim=0, standardize=False, norm=False, Eyes=None, pool=None, parallel=False,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.obs_shape = getattr(obs_spec, 'shape', obs_spec)  # Allow spec or shape

        for key in ('mean', 'stddev', 'low', 'high'):
            setattr(self, key, None if getattr(obs_spec, key, None) is None else torch.as_tensor(obs_spec[key]))

        self.standardize = \
            standardize and None not in [self.mean, self.stddev]  # Whether to center-scale (0 mean, 1 stddev)
        self.normalize = norm and None not in [self.low, self.high]  # Whether to shift-max scale

        self.norm = norm  # Can control normalization range, default: [-1, 1]

        # Dimensions
        obs_shape = [*(1,) * (len(self.obs_shape) < 2), *self.obs_shape]  # Create at least 1 channel dim & spatial dim
        obs_shape[0] += context_dim

        self.Eyes = instantiate(Eyes, **Utils.adaptive_shaping(obs_shape)) or CNN(obs_shape)

        adapt_cnn(self.Eyes, obs_shape)  # Adapt 2d CNN kernel sizes for 1d or small-d compatibility

        self._eyes = self.Eyes  # Pre-construct uses act method hidden in Parallel

        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.feature_shape = Utils.repr_shape(obs_shape, self.Eyes)  # Feature map shape

        self.pool = instantiate(pool, **Utils.adaptive_shaping(self.feature_shape)) or nn.Flatten()  # TODO Identity?

        self.repr_shape = Utils.repr_shape(self.feature_shape, self.pool)  # Shape after pooling

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False).eval()

    def forward(self, obs, *context, pool=True):
        # Operate on non-batch dims, then restore

        dims = len(self.obs_shape)

        batch_dims = obs.shape[:max(1, len(obs.shape) - dims)]  # Preserve leading dims
        axes = (1,) * (dims - 1)  # Spatial axes, useful for dynamic input shapes

        # Standardize/normalize pixels
        if self.standardize:
            obs = (obs - self.mean.to(obs.device).view(-1, *axes)) / self.stddev.to(obs.device).view(-1, *axes)
        elif self.normalize:
            obs = (self.norm * 2) * (obs - self.low) / (self.high - self.low) - float(self.norm)

        try:
            channel_dim = (1,) * (not axes)  # At least 1 channel dim and spatial dim
            obs = obs.reshape(-1, *channel_dim, *self.obs_shape)  # Validate shape, collapse batch dims
        except RuntimeError:
            raise RuntimeError('\nObs shape does not broadcast to pre-defined obs shape '
                               f'{tuple(obs.shape[1:])}, ≠ {self.obs_shape}')

        # Optionally append a 1D context to channels, broadcasting
        if context:
            obs = torch.cat([obs, *[c.reshape(obs.shape[0], c.shape[-1], *axes or (1,)).expand(-1, -1, *obs.shape[2:])
                                    for c in context]], 1)

        # CNN encode
        h = self.Eyes(obs)

        try:
            h = h.view(h.shape[0], *self.feature_shape)  # Validate shape
        except RuntimeError:
            raise RuntimeError('\nFeature shape cannot broadcast to pre-computed feature_shape '
                               f'{tuple(h.shape[1:])}≠{self.feature_shape}')

        if pool:
            h = self.pool(h)
            try:
                h = h.view(h.shape[0], *self.repr_shape)  # Validate shape
            except RuntimeError:
                raise RuntimeError('\nOutput shape after pooling does not match pre-computed repr_shape '
                                   f'{tuple(h.shape[1:])}≠{self.repr_shape}')

        h = h.view(*batch_dims, *h.shape[1:])  # Restore leading dims
        return h


# Adaptive Eyes
def adapt_cnn(block, obs_shape):
    """
    Adapts a 2d CNN to a smaller dimensionality or truncates adaptively (in case an image's spatial dim < kernel size)
    """
    name = type(block).__name__
    Nd = 2 if '2d' in name else 1 if '1d' in name else 0

    if Nd:
        # Set attributes of block adaptively according to obs shape
        for attr in ['kernel_size', 'padding', 'stride', 'dilation', 'output_padding', 'output_size']:
            if hasattr(block, attr):
                val = getattr(nn.modules.conv, '_single' if Nd < 2 else '_pair')(getattr(block, attr))  # To tuple
                if isinstance(val[0], int):
                    setattr(block, attr, tuple(adapt if 'Adaptive' in name else min(dim, adapt)
                                               for dim, adapt in zip(val, obs_shape[1:])))  # Truncate

        # Update 2d operation to 1d if needed
        if len(obs_shape) < Nd + 1:
            block.forward = getattr(nn, name.replace('2d', '1d')).forward.__get__(block)

            # Contract
            if isinstance(block, (nn.Conv2d, nn.ConvTranspose2d)):
                block.weight = nn.Parameter(block.weight[:, :, :, 0])
                replace = nn.Conv1d if isinstance(block, nn.Conv2d) else nn.ConvTranspose1d
                block._conv_forward = replace._conv_forward.__get__(block, type(block))

        # Truncate
        if hasattr(block, '_conv_forward'):
            block.weight = nn.Parameter(block.weight[:, :, :block.kernel_size[0]] if len(obs_shape) < 3
                                        else block.weight[:, :, :block.kernel_size[0], :block.kernel_size[1]])
        elif hasattr(block, '_check_input_dim'):
            block._check_input_dim = lambda *_: None
    elif hasattr(block, 'modules') and name != 'TIMM':
        for layer in block.children():
            # Iterate through all layers
            adapt_cnn(layer, obs_shape)  # Dimensionality-adaptivity
            # Account for multiple streams in Residual
            if name != 'Residual' or block.down_sample is None or layer == block.down_sample:
                obs_shape = Utils.repr_shape(obs_shape, layer)  # Update shape
