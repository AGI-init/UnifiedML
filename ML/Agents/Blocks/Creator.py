# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
from functools import cached_property
import copy

import torch
from torch import nn

from minihydra import instantiate

from Agents.Blocks.Architectures.Distributions import TruncatedNormal, NormalizedCategorical

import Utils


class Creator(nn.Module):
    """Creates a policy distribution for sampling actions and ensembles and computing probabilistic measures."""
    def __init__(self, action_spec, discrete=False, rand_steps=0, temp_schedule=None, policy=None, ActionExtractor=None,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_spec = action_spec

        self.policy = policy  # Exploration and exploitation policy recipe

        self.rand_steps = rand_steps

        # Entropy temperature for sampling
        self.temp_schedule = temp_schedule

        # A mapping that can be applied after or concurrently with action sampling
        self.ActionExtractor = instantiate(ActionExtractor, **Utils.adaptive_shaping(math.prod(self.action_spec.shape))
                                           ) or nn.Identity()

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False).eval()

    # Creates actor policy Pi
    def Omega(self, mean, stddev, step=1):

        r = step < self.rand_steps  # Whether to sample uniformly random actions

        # Optionally create policy from recipe
        return instantiate(self.policy, mean=mean, stddev=stddev, step=step,
                           action_spec=self.action_spec, discrete=self.discrete,
                           rand=r, temp_schedule=self.temp_schedule, ActionExtractor=self.ActionExtractor) or \
            MonteCarlo(self.action_spec, self.discrete, mean, stddev, step, r, self.temp_schedule, self.ActionExtractor)


# Policy
class MonteCarlo:
    """Exploration and exploitation policy distribution compatible with discrete and continuous spaces and ensembles."""
    def __init__(self, action_spec, discrete, mean, stddev, step=1, rand=False, temp_schedule=1, ActionExtractor=None):

        self.discrete = discrete
        self.discrete_as_continuous = action_spec.discrete and not self.discrete

        self.low, self.high = action_spec.low, action_spec.high

        self.step = step

        if self.discrete:
            self.All_Qs = mean  # [b, e, n, d]
        else:
            # Normalized  TODO perhaps normalize after sampling for stddev consistency across scales (DrQV2 to best)
            self.mean = mean if not (self.low or self.high) else mean.tanh() if self.low == -1 and self.high == 1 \
                else (mean.tanh() + 1) / 2 * (self.high - self.low) + self.low  # [b, e, n, d]

        # Randomness
        self.stddev = stddev
        self.rand = rand
        # TODO
        # self.rand_clip = 1
        # self.stddev_clip = 0.3  # TODO Should include in Categorical?
        # Or self.dev = None  # Can directly set the deviation  Note: Discrete uses dev then defaults to stddev

        # A secondary mapping that is applied after sampling an continuous-action
        self.ActionExtractor = ActionExtractor or nn.Identity()

        self.store = None

        if self.discrete_as_continuous:
            self.temp = Utils.schedule(temp_schedule, self.step)  # Temp for controlling entropy of re-sample

    # Sub policy
    @cached_property
    def Psi(self):
        if self.discrete:
            # Pessimistic Q-values per action
            logits, ind = self.All_Qs.min(1)  # Min-reduced ensemble [b, n, d]

            # Corresponding entropy temperature
            stddev = self.stddev if isinstance(self.stddev, float) or self.stddev.nelement() == 1 \
                else Utils.gather(self.stddev, ind.unsqueeze(1), 1, 1).flatten(1).sum(1)  # Min-reduced ensemble std [b]

            try:
                return NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
            except ValueError:
                # M1 Mac MPS can sometimes throw ValueError on log-sum-exp trick
                Psi = NormalizedCategorical(logits=logits.to('cpu'), low=self.low, high=self.high, temp=stddev, dim=-2)
                Psi.logits = Psi.logits.to(logits.device)
                return Psi
        else:
            return TruncatedNormal(self.mean, self.stddev, low=self.low, high=self.high, stddev_clip=0.3)

    def log_prob(self, action=None):  # (Log-space is more numerically stable)
        # Log-probability
        log_prob = self.Psi.log_prob(action)  # [b, n', d] if discrete, [b, e*n', 1, d] if continuous

        # If continuous-action is a discrete distribution, it gets double-sampled  TODO logsumexp
        if self.discrete_as_continuous:
            log_prob = (log_prob + action / self.temp).sum(-1)  # (Adding log-probs is equivalent to multiplying probs)

        return log_prob.sum(-1).flatten(1)  # [b, e*n']

    @cached_property
    def _entropy(self):
        return self.Psi.entropy() if self.discrete else self.Psi.entropy().mean((2, 3))  # [b, e]

    def entropy(self):
        # If continuous-action is a discrete distribution, 2nd sample also has entropy
        if self.discrete_as_continuous:
            # Approximate joint entropy  TODO NormalizedCategorical which logsumexps first then weighs temp
            return self._entropy + torch.distributions.Categorical(logits=self.mean.mean(-1) / self.temp).entropy()

        return self._entropy  # [b, e]

    # Exploration policy
    def sample(self, sample_shape=None, detach=True):
        # Monte Carlo

        # Sample
        action = self.Psi.sample(sample_shape or 1) if detach else self.Psi.rsample(sample_shape or 1)

        # Can optionally map a continuous-action after sampling
        if not self.discrete:
            action = self.ActionExtractor(action)

        # Reduce continuous-action ensemble
        if sample_shape is None and action.shape[1] > 1 and not self.discrete:
            return action[:, torch.randint(action.shape[1], [])]  # Uniform sample again across ensemble

        # Take random action  (Note: Due to rand_steps duration usually being short, efficiency is not optimized)
        if self.rand:
            action = action.float().uniform_(self.low, self.high)

        # If sampled action is a discrete distribution, sample again  TODO Discrete norm?
        if sample_shape is None and self.discrete_as_continuous:
            self.store = action  # TODO NormalizedCategorical which logsumexps first then weighs temp
            action = torch.distributions.Categorical(logits=action.movedim(-2, -1) / self.temp).sample()  # Sample again

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
    @cached_property
    def best(self):
        # Absolute Determinism (kind of)

        # Argmax for discrete, extract action for continuous  (Note: Ensemble reduce somewhat random; could use critic)
        action = self.Psi.normalize(self.Psi.logits.argmax(-1, keepdim=True).transpose(-1, -2)) if self.discrete \
            else self.ActionExtractor(self.mean[:, torch.randint(self.mean.shape[1], [])])  # Extract ensemble-reduce

        # If continuous-action is a discrete distribution  TODO Discrete norm? Also, can do env-rounding here if needed
        if self.discrete_as_continuous:
            self.store = action
            action = action.argmax(1)  # Argmax

        return action

    # TODO
    def judgement(self, obs, action, critic):
        # Pessimism
        return critic(obs, action).min(1)[0]  # Reduce critic ensemble pessimistically
