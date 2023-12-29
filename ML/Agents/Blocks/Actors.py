# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

from torch import nn

from minihydra import instantiate

from Agents.Blocks.Architectures.Ensemble import Ensemble
from Agents.Blocks.Architectures.MLP import MLP

import Utils

from Agents.Blocks.Creator import Creator


class EnsemblePiActor(nn.Module):
    """Ensemble of Gaussian or Categorical policies Pi, generalized for discrete or continuous action spaces."""
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_spec, trunk=None, Pi_head=None, ensemble_size=1,
                 discrete=False, stddev_schedule=1, creator=None, rand_steps=0, parallel=False,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.num_actions = action_spec.discrete_bins or 1  # n
        self.action_dim = math.prod(action_spec.shape) * (1 if stddev_schedule else 2)  # d, or d * 2

        # Standard deviation ("Uncertainty" / "Randomness") scheduler for exploration A.K.A. entropy temperature
        self.stddev_schedule = stddev_schedule

        in_dim = math.prod(repr_shape)

        self.trunk = instantiate(trunk, **Utils.adaptive_shaping(repr_shape, [trunk_dim])) or nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())

        in_shape = Utils.repr_shape(repr_shape, self.trunk)  # Will be trunk_dim when possible
        out_shape = [self.num_actions * action_spec.shape[0] * (1 if stddev_schedule else 2), *action_spec.shape[1:]]

        # Ensemble
        self.Pi_head = Ensemble([instantiate(Pi_head, i, **Utils.adaptive_shaping(in_shape, out_shape))
                                 or MLP(in_shape, out_shape, hidden_dim, 2) for i in range(ensemble_size)])

        self._pi_head = self.Pi_head  # Pre-construct uses act method hidden in Parallel

        if parallel:
            self.Pi_head = nn.DataParallel(self.Pi_head)  # Parallel on visible GPUs

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False).eval()

        # Can create the policy distribution
        self.creator = Creator(action_spec, self.discrete, rand_steps, **creator or {},
                               lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay, ema_decay=ema_decay)
        if ema_decay:
            setattr(self.ema, 'creator', self.creator.ema)  # Creator EMA

    def forward(self, obs, step=1):

        h = self.trunk(obs)

        # Action data or "Belief"
        mean = self.Pi_head(h).view(h.shape[0], -1, self.num_actions, self.action_dim)  # [b, e, n, d or 2 * d]

        if self.stddev_schedule is None:
            mean, log_stddev = mean.chunk(2, dim=-1)  # [b, e, n, d]

            # "Uncertainty"
            stddev = log_stddev.exp()  # [b, e, n, d]  # Learnable entropy temperature
        else:
            # "Uncertainty"
            stddev = Utils.schedule(self.stddev_schedule, step)  # Scalar entropy temperature

        # Returns policy distribution Pi
        return self.creator.train(self.training).Omega(mean, stddev, step)  # Creates policy distribution Pi
