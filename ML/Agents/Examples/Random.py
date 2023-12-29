# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import time

import torch

import Utils


class RandomAgent(torch.nn.Module):
    """Random Agent generalized to support everything"""
    def __init__(self,
                 obs_spec, action_spec, generate
                 ):
        super().__init__()

        action_dim = math.prod(obs_spec.shape) if generate else action_spec.shape[-1]  # Support generative shapes

        self.actor = Utils.Rand(action_dim, uniform=True)  # Output random numbers

    def act(self, obs):

        action = self.actor(obs) * 2 - 1  # [-1, 1]

        return action

    def learn(self, replay=None):
        return
