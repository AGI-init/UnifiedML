# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import warnings

import torch
from torch.nn.functional import cross_entropy, mse_loss

from minihydra import instantiate

from Agents.Blocks.Architectures import MLP, MiniResNet

import Utils

from Agents.Blocks.Augmentations import RandomShiftsAug
from Agents.Blocks.Encoders import CNNEncoder
from Agents.Blocks.Actors import EnsemblePiActor
from Agents.Blocks.Critics import EnsembleQCritic

from Agents.Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class Agent(torch.nn.Module):
    """Actor Critic Creator (AC2)
    RL, classification, regression, generative mode; online, offline; self-supervised learning; critic/actor ensembles;
    action space conversions; optimization schedules; EMA"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 rand_steps, stddev_schedule,  # Exploration
                 discrete, RL, supervise, generate, parallel, log,  # On-boarding
                 num_critics, num_actors, depth  # Ensembles & self-supervision
                 ):
        super().__init__()

        self.discrete = discrete and not generate  # Discrete & Continuous supported!
        self.supervise = supervise  # And classification...
        self.classify = action_spec.discrete  # Regression
        self.RL = RL or generate  # RL,
        self.generate = generate  # And generative modeling, too

        self.log = log

        self.num_actors = max(num_critics, num_actors) if self.discrete and self.RL else num_actors
        self.depth = depth  # Dynamics prediction depth

        # Image augmentation
        self.aug = Utils.Modals(instantiate(recipes.aug) or RandomShiftsAug(pad=4))\

        # RL -> generate conversion
        if self.generate:
            standardize = False
            norm = True  # Normalize Obs to range [-1, 1]
            rand_steps = 0

            # Action = Imagined Obs
            action_spec.update({'shape': obs_spec.shape, 'discrete_bins': None,
                                'low': -1, 'high': 1, 'discrete': False})

            # Remove encoder, replace trunk with random noise
            recipes.encoder.Eyes = torch.nn.Identity()  # Generate "imagines" — no need for " seeing " via Eyes
            recipes.actor.trunk = Utils.Rand(size=trunk_dim)  # Generator observes random Gaussian noise as input

        self.discrete_as_continuous = action_spec.discrete and not self.discrete

        # Discrete -> continuous conversion
        if self.discrete_as_continuous:
            # Normalizing actions to range [-1, 1] significantly helps continuous RL
            action_spec.low, action_spec.high = (-1, 1) if self.RL else (None, None)

        # Continuous -> discrete conversion
        if self.discrete and not action_spec.discrete:
            assert num_actions > 1, 'Num actions cannot be 1 when discrete; try the "num_actions=" flag (>1) to ' \
                                    'divide each action dimension into discrete bins, or specify "discrete=false".'

            action_spec.discrete_bins = num_actions  # Continuous env has no discrete bins by default, must specify

        # Note: Slower unnecessary EMA updates when not RL or EMA
        self.encoder = CNNEncoder(obs_spec, standardize=standardize, norm=norm, **recipes.encoder, parallel=parallel,
                                  lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                  ema_decay=ema_decay * (RL and not generate and depth or ema))

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                     ensemble_size=self.num_actors, discrete=self.discrete, parallel=parallel,
                                     stddev_schedule=stddev_schedule, creator=recipes.creator, rand_steps=rand_steps,
                                     lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                     ema_decay=ema_decay * ema)

        # Dynamics
        if self.depth and not self.generate:
            shape = list(self.encoder.feature_shape)

            # Action -> One-Hot, if single-dim discrete, otherwise action shape
            self.action_dim = action_spec.discrete_bins if self.discrete and action_spec.shape == (1,) \
                else self.actor.num_actions * self.actor.action_dim if self.discrete_as_continuous \
                else self.actor.action_dim

            shape[0] += self.action_dim  # Predicting from obs and action

            resnet = MiniResNet(input_shape=shape, stride=1, dims=(32, self.encoder.feature_shape[0]), depths=(1,))

            self.dynamics = CNNEncoder(self.encoder.feature_shape, context_dim=self.action_dim,
                                       Eyes=resnet, parallel=parallel,
                                       lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay)

            # Self supervisors
            self.projector = CNNEncoder(self.encoder.feature_shape,
                                        Eyes=MLP(self.encoder.feature_shape, hidden_dim, hidden_dim, 2),
                                        lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                        ema_decay=ema_decay)

            self.predictor = CNNEncoder(self.projector.repr_shape,
                                        Eyes=MLP(self.projector.repr_shape, hidden_dim, hidden_dim, 2),
                                        lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay)

        if self.RL:
            # When discrete, Critic <- Actor
            if self.discrete:
                recipes.critic.trunk = self.actor.trunk
                recipes.critic.Q_head = self.actor.Pi_head.ensemble

            self.critic = EnsembleQCritic(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                          ensemble_size=self.num_actors if self.discrete else num_critics,
                                          discrete=self.discrete, ignore_obs=self.generate,
                                          lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                          ema_decay=ema_decay * (RL and not generate or ema))

        # "Birth"

    def act(self, obs, store):
        # "See"
        obs = self.encoder(obs)

        # Act
        Pi = self.actor(obs, self.step)

        action = Pi.sample() if self.training \
            else Pi.best

        # Creator may store distribution as action rather than sampled action
        if Pi.store is not None:
            store.action = Pi.store.cpu().numpy()

        return action

    def learn(self, replay, log):
        if not self.log:
            log = None

        # "Recall"

        if self.depth > 0:
            replay.include_trajectories()

        batch = next(replay)

        # "Perceive"

        # Augment, encode present
        batch = self.aug(batch)
        features = self.encoder(batch.obs, pool=False)
        batch.obs = self.encoder.pool(features)

        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                # Augment, encode future
                batch.next_obs = self.aug(batch.next_obs)
                batch.next_obs = self.encoder(batch.next_obs)

        # "Begin acquisition of Wisdom"

        instruct = not self.generate and 'label' in batch

        # Classification & regression
        if (self.supervise or replay.offline) and instruct:
            # "Via Example" / "Parental Support" / "School"

            # Inference
            Pi = self.actor(batch.obs)
            y_predicted = (Pi.All_Qs if self.discrete else Pi.mean).mean(1)  # Average over ensembles

            # Cross entropy error: classification
            if self.classify:
                batch.label = batch.label.view(len(y_predicted), -1)

                error = cross_entropy(y_predicted, batch.label.long(),
                                      reduction='none' if self.RL and replay.offline else 'mean')

                # Accuracy computation
                if self.log or self.RL and replay.offline:
                    index = y_predicted.argmax(1, keepdim=True)  # Predicted class
                    correct = (index.squeeze(1) == batch.label).float()
                    accuracy = correct.mean()

                    if self.log:
                        log.update({'accuracy': accuracy})
            # Mean-squared error: regression
            else:
                error = mse_loss(y_predicted, batch.label.view_as(y_predicted),
                                 reduction='none' if self.RL and replay.offline else 'mean')

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = error.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, epoch=self.epoch if replay.offline else self.episode, retain_graph=True)

                if self.log:
                    log.update({'supervised_loss': supervised_loss})

        # Reinforcement learning / generative modeling
        if self.RL:

            # Action and reward for supervised reinforcement learning
            if instruct:
                # "Via Feedback" / "Test Score" / "Letter Grade"

                if replay.offline:
                    batch.action = (index if self.discrete and self.classify else y_predicted).detach()
                    batch.reward = correct if self.discrete and self.classify else -error.detach()  # reward = -error
                else:  # Use Replay action from Online training
                    # reward = -error  TODO Check if label shape still correct
                    #                       batch.label = batch.label.view(len(y_predicted), -1)
                    batch.reward = (batch.action.squeeze(1) == batch.label).float() if self.discrete \
                        else -cross_entropy(batch.action.squeeze(1), batch.label.long(), reduction='none')

            # Generative modeling
            if self.generate:
                # "Imagine"

                batch.action = batch.obs
                batch.reward = torch.ones(len(batch.obs), 1, device=batch.obs.device)  # Discriminate Real

                # Critic loss
                critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                          batch.obs, batch.action, batch.reward)

                # Update discriminator
                Utils.optimize(critic_loss, self.critic, epoch=self.epoch if replay.offline else self.episode)

                batch.next_obs = None

                Pi = self.actor(batch.obs)
                generated_image = Pi.best.flatten(1)  # Imagined image

                batch.action, batch.reward = generated_image, torch.zeros_like(batch.reward)  # Discriminate Fake

            # Update reward log
            if self.log:
                log.update({'reward': batch.reward})

            # "Discern" / "Discriminate"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor, batch.obs, batch.action, batch.reward,
                                                      batch.discount, getattr(batch, 'next_obs', None),
                                                      self.step, log=log)

            # "Foretell"

            # Can only predict dynamics from available trajectories
            if self.depth > replay.nstep:
                warnings.warn(f"Dynamics 'depth' cannot exceed trajectory 'nstep'. Lowering 'depth' to {replay.nstep}. "
                              f"You can increase 'nstep' with the 'nstep={self.depth}' flag.")
                self.depth = replay.nstep

            # Dynamics loss
            dynamics_loss = 0 if self.depth == 0 or self.generate \
                else SelfSupervisedLearning.dynamicsLearning(features, batch.traj_o, batch.traj_a, batch.traj_r,
                                                             self.encoder, self.dynamics, self.projector, self.predictor,
                                                             depth=self.depth, action_dim=self.action_dim, log=log)

            models = () if self.generate or not self.depth else (self.dynamics, self.projector, self.predictor)

            # "Sharpen Foresight"

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss, self.critic, *models, retain_graph=self.generate,
                           epoch=self.epoch if replay.offline else self.episode)

        # Update encoder
        Utils.optimize(None,  # Using gradients from previous losses
                       self.encoder, epoch=self.epoch if replay.offline else self.episode)

        if self.RL and not self.discrete:
            # "Change, Grow,  Ascend"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, batch.obs.detach(), batch.action,
                                                           not self.generate, self.step, log=log)

            # Update actor
            Utils.optimize(actor_loss, self.actor, epoch=self.epoch if replay.offline else self.episode)
