# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy


def ensembleQLearning(critic, actor, obs, action, reward, discount=1, next_obs=None, step=0, log=None):
    # Non-empty next_obs
    has_future = next_obs is not None and bool(next_obs.nelement())

    # Compute Bellman target
    with torch.no_grad():
        # Current reward
        target_Q = reward

        # Discounted future reward
        if has_future:
            # Get actions for next_obs
            next_Pi = actor(next_obs, step)

            # Discrete Critic tabulates all actions for single-dim discrete envs a priori, no need to sample
            next_action = None if critic.all_actions_known else next_Pi.sample(1)  # Sample

            # Discrete Actor already computed Q-values and they're already known by Policy
            All_Next_Qs = next_Pi.All_Qs if actor.discrete else None

            # Q-values per action
            next_Qs = critic.ema.eval()(next_obs, next_action, All_Next_Qs)  # Call a delayed-copy Critic: Q(obs, a)

            # Pessimistic Q-values per action
            next_q = next_Qs.min(1)[0]  # Min-reduced critic ensemble

            # Weigh each action's pessimistic Q-value by its probability
            next_action_prob = next_q.size(1) < 2 or next_Pi.log_prob(next_action).softmax(-1)  # Action probability
            next_v = next_q * next_action_prob
            next_v = next_v.squeeze(-1) if next_q.size(1) < 2 else next_v.sum(-1)   # Expected Q-value = E_a[Q(obs, a)]

            target_Q += discount * next_v  # Add expected future discounted-cumulative-reward to reward

    Qs = critic(obs, action)  # Q-ensemble

    # Use BCE if Critic is Sigmoid-activated, else MSE
    criterion = binary_cross_entropy if critic.binary else mse_loss

    # Temporal difference (TD) error
    # q_loss = criterion(Qs.float(), target_Q.view(-1, 1, 1).float().expand_as(Qs))
    q_loss = criterion(Qs, target_Q.view(-1, 1, 1).expand_as(Qs))  # TODO Replay set float if MP

    if log is not None:
        log['temporal_difference_error'] = q_loss
        log.update({f'q{i}': Qs[:, i].to('cpu' if Qs.device.type == 'mps' else Qs).median()  # median not on MPS
                    for i in range(Qs.shape[1])})
        log['target_q'] = target_Q.mean()

    return q_loss
