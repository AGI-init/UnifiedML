# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents.Blocks.Augmentations import RandomShiftsAug
from Agents.Blocks.Encoders import CNNEncoder
from Agents.Blocks.Actors import EnsemblePiActor
from Agents.Blocks.Critics import EnsembleQCritic

from Agents.Losses import QLearning, PolicyLearning


class DrQV2Agent(torch.nn.Module):
    """Data Regularized Q-Learning version 2 (https://arxiv.org/abs/2107.09645)
    State of the art reinforcement learner"""
    def __init__(self,
                 obs_spec, action_spec, trunk_dim, hidden_dim,  # Architecture
                 lr, ema_decay,  # Optimization
                 rand_steps, stddev_schedule,  # Exploration
                 log,  # On-boarding
                 ):
        super().__init__()

        self.aug = RandomShiftsAug(pad=4)

        self.encoder = CNNEncoder(obs_spec, norm=0.5, lr=lr)

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec,
                                     stddev_schedule=stddev_schedule, rand_steps=rand_steps, lr=lr)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec,
                                      lr=lr, ema_decay=ema_decay)

        self.log = log

    def act(self, obs):
        obs = self.encoder(obs)
        Pi = self.actor(obs, self.step)
        action = Pi.sample() if self.training else Pi.best
        return action

    def learn(self, replay, log):
        if not self.log:
            log = None

        batch = next(replay)

        # Augment, encode present
        batch.obs = self.aug(batch.obs)
        batch.obs = self.encoder(batch.obs)

        if replay.nstep:
            with torch.no_grad():
                # Augment, encode future
                batch.next_obs = self.aug(batch.next_obs)
                batch.next_obs = self.encoder(batch.next_obs)

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.critic, self.actor, batch.obs, batch.action, batch.reward,
                                                  batch.discount, batch.next_obs, self.step, log=log)

        # Update encoder and critic
        Utils.optimize(critic_loss, self.encoder, self.critic)

        # Actor loss
        actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, batch.obs.detach(),
                                                       step=self.step, log=log)

        # Update actor
        Utils.optimize(actor_loss, self.actor)
