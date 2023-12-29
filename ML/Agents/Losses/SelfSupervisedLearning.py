# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


def bootstrapYourOwnLatent(obs, positive, encoder, projector, predictor, log=None):
    """
    Bootstrap Your Own Latent (https://arxiv.org/abs/2006.07733),
    Self-supervision via EMA
    """
    with torch.no_grad():
        positive = encoder.ema(positive, pool=False)
        positive = projector.ema(positive)

    # Assumes obs already encoded
    anchor = predictor(projector(obs))

    self_supervised_loss = -F.cosine_similarity(anchor, positive, -1).mean()

    if log is not None:
        log['byol_loss'] = self_supervised_loss

    return self_supervised_loss


def dynamicsLearning(obs, traj_o, traj_a, traj_r,
                     encoder, dynamics, projector, obs_predictor=None, reward_predictor=None,
                     depth=1, action_dim=0, log=None):
    assert depth < traj_o.shape[1], f"Depth {depth} exceeds future trajectory size of {traj_o.shape[1] - 1} steps"

    # Dynamics accepts a single flat action
    traj_a = traj_a.flatten(2)

    # If discrete action, converts to one-hot
    if traj_a.shape[-1] == 1 and action_dim > 1:
        traj_a = Utils.one_hot(traj_a, num_classes=action_dim)

    # Predict future
    forecast = [dynamics(obs, traj_a[:, 0], pool=False)]
    for k in range(1, depth):
        forecast.append(dynamics(forecast[-1], traj_a[:, k], pool=False))
    forecast = torch.stack(forecast, 1)

    # Self supervision
    dynamics_loss = 0
    future = traj_o[:, 1:depth + 1]
    if obs_predictor is not None:
        dynamics_loss -= bootstrapYourOwnLatent(forecast, future, encoder, projector, obs_predictor, log)

    if reward_predictor is not None:
        reward_prediction = reward_predictor(forecast)
        dynamics_loss -= F.mse_loss(reward_prediction, traj_r[:, :depth])

    return dynamics_loss
