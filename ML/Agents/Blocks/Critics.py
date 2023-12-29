# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn

from minihydra import instantiate

from Agents.Blocks.Architectures.Ensemble import Ensemble
from Agents.Blocks.Architectures.MLP import MLP

import Utils


class EnsembleQCritic(nn.Module):
    """Ensemble Q-learning, generalized for discrete or continuous action spaces."""
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_spec, trunk=None, Q_head=None, ensemble_size=2,
                 discrete=False, ignore_obs=False, optim=None, scheduler=None, lr=None, lr_decay_epochs=None,
                 weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.num_actions = action_spec.discrete_bins or 1  # n
        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        self.ignore_obs = ignore_obs and not discrete  # Discrete critic always requires observation

        self.all_actions_known = self.discrete and self.action_dim == 1  # Whether Q-values correspond to all actions

        in_dim = math.prod(repr_shape)
        out_shape = [self.num_actions, *action_spec.shape] if discrete else [1]

        self.trunk = instantiate(trunk, **Utils.adaptive_shaping(repr_shape, [trunk_dim])) or nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())  # Not used if ignore obs!

        # Continuous action-space Critic gets (obs, action) as input
        in_shape = action_spec.shape if ignore_obs else [trunk_dim + (0 if discrete  # Not dynamic
                                                                      else self.num_actions * self.action_dim)]

        # Ensemble
        self.Q_head = Ensemble([instantiate(Q_head, i, **Utils.adaptive_shaping(in_shape, out_shape)) or
                                MLP(in_shape, out_shape, hidden_dim, 2) for i in range(ensemble_size)])  # e

        self.binary = isinstance(tuple(self.Q_head.modules())[-1],
                                 nn.Sigmoid)  # Whether Sigmoid-activated e.g. in GANs

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False).eval()

    def forward(self, obs, action=None, All_Qs=None):

        h = torch.empty((action.shape[0], 0), device=action.device) if self.ignore_obs \
            else self.trunk(obs)

        assert self.all_actions_known or action is not None, 'Action is needed by critic.'

        batch_size = h.shape[0]

        if self.discrete:
            if All_Qs is None:
                # All actions' Q-values
                All_Qs = self.Q_head(h).view(batch_size, -1, self.num_actions, self.action_dim)  # [b, e, n, d]

            if action is None:
                # Return Q-values for all actions
                Qs = All_Qs.squeeze(-1)  # [b, e, n]
            else:
                action = action.view(action.shape[0], 1, -1, self.action_dim)  # [b, 1, n', d]

                # Q values for discrete action(s)
                Qs = Utils.gather(All_Qs, self.to_indices(action), -2, -2).mean(-1)  # [b, e, n']
        else:
            action = action.reshape(batch_size, -1, self.num_actions * self.action_dim)  # [b, n', n * d]

            h = h.unsqueeze(1).expand(*action.shape[:-1], -1)  # One for each action

            # Q-values for continuous action(s)
            Qs = self.Q_head(h, action).squeeze(-1)  # [b, e, n']

        return Qs

    def to_indices(self, action):  # Same as un-normalize
        # Action to indices
        if None in (self.low, self.high) or (self.low, self.high) == (0, self.num_actions - 1):
            return action

        return (action - self.low) / (self.high - self.low) * (self.num_actions - 1)  # Inverse of low/high normalize
