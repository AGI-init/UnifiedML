# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.


def deepPolicyGradient(actor, critic, obs, action=None, sample=True, step=1, log=None):

    if action is None or not action.requires_grad:  # If None or not differentiable
        Pi = actor(obs, step)
        action = Pi.rsample() if sample else Pi.mean  # Differentiable action ensemble

        if Pi.store is not None:
            action = Pi.store  # Creator might learn a different action from the one actually sent to Env

    Qs = critic(obs, action)

    q, _ = Qs.min(1)  # Min-reduced critic ensemble

    # When Sigmoid-activated
    if critic.binary:
        q = q.log()

    # Policy gradient ascent
    policy_loss = -q.mean()

    if log is not None:
        log['policy_loss'] = policy_loss

    return policy_loss
