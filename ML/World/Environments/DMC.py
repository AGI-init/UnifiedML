# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import warnings
from collections import deque

from dm_env import StepType

import numpy as np

from minihydra import Args


class DMC:
    """
    A general-purpose environment must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "obs_spec" attribute which includes:
        - "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (4) "action-spec" attribute which includes:
        - "shape", "discrete_bins" (should be None if not discrete), "low", "high", and "discrete"

    An "exp" (experience) is a dict consisting of keys such as "obs", "action", "reward", and "label".

    ---

    Can optionally include a frame_stack, action_repeat method.

    """
    def __init__(self, task='cheetah_run', seed=0, frame_stack=3, action_repeat=2):
        self.episode_done = False

        # Make env

        # Import DM Control here to avoid glfw warnings

        try:
            # Try EGL rendering (faster)
            os.environ['MUJOCO_GL'] = 'egl'
            from dm_control import manipulation, suite
        except ImportError:
            del os.environ['MUJOCO_GL']  # Otherwise GLFW
            from dm_control import manipulation, suite

        from dm_control.suite.wrappers import action_scale, pixels

        warnings.filterwarnings("ignore", message='.* is deprecated and will be removed in Pillow 10')

        domain, task = task.split('_', 1)
        domain = 'ball_in_cup' if domain == 'cup' else domain  # Overwrite cup to ball_in_cup

        # Load task
        if (domain, task) in suite.ALL_TASKS:
            self.env = suite.load(domain,
                                  task,
                                  task_kwargs={'random': seed},
                                  visualize_reward=False)  # Don't visualize reward
            self.key = 'pixels'
        else:
            task = f'{domain}_{task}_vision'
            self.env = manipulation.load(task, seed=seed)
            self.key = 'front_close'

        # Rescale actions to range [-1, 1]
        self.env = action_scale.Wrapper(self.env, minimum=-1.0, maximum=+1.0)

        # Add rendering for classical tasks
        if (domain, task) in suite.ALL_TASKS:
            # Zoom in camera for quadruped
            camera_id = dict(quadruped=2).get(domain, 0)
            render_kwargs = dict(height=84, width=84, camera_id=camera_id)
            self.env = pixels.Wrapper(self.env,
                                      pixels_only=True,  # No proprioception (key <- 'position')
                                      render_kwargs=render_kwargs)  # Should be applied after action repeat!

        self.obs_spec = {'low': 0, 'high': 255}
        self.action_spec = {'shape': self.env.action_spec().shape, 'low': -1, 'high': 1}

        self.action_repeat = action_repeat  # action_repeat attribute
        self.frames = deque([], frame_stack or 1)  # For frame_stack method

    def step(self, action):
        # Remove batch dim
        action = np.reshape(action, self.action_spec['shape'])

        # Step env
        reward = np.zeros([])
        for _ in range(self.action_repeat):
            # time_step = self.env.step(action)
            time_step = self.env._env.step(action)  # Don't render yet
            reward += time_step.reward
            self.episode_done = time_step.step_type == StepType.LAST
            if self.episode_done:
                break

        time_step = self.env._add_pixel_observation(time_step)  # Render pixels

        obs = time_step.observation[self.key].copy()  # DMC returns numpy arrays with negative strides, need to copy

        # Add batch dim
        obs = np.expand_dims(obs, 0)
        # Channel-first
        obs = obs.transpose(0, 3, 1, 2)

        prev = {'reward': reward}  # Reward for previous action
        now = {'obs': obs, 'done': self.episode_done}  # New state

        return prev, now

    def frame_stack(self, obs):
        if self.frames.maxlen == 1:
            return obs

        self.frames.extend([obs] * (self.frames.maxlen - len(self.frames) + 1))
        return np.concatenate(list(self.frames), axis=1)

    def reset(self):
        time_step = self.env.reset()
        self.episode_done = False

        obs = time_step.observation[self.key].copy()  # DMC returns numpy arrays with negative strides, need to copy

        # Create experience
        exp = {'obs': obs, 'reward': time_step.reward, 'done': False}  # TODO Auto-add Done
        # Add batch dim
        exp['obs'] = np.expand_dims(exp['obs'], 0)
        # Channel-first
        exp['obs'] = exp['obs'].transpose(0, 3, 1, 2)

        # Reset frame stack
        self.frames.clear()

        return Args(exp)  # Experience

    def render(self):
        return self.env.physics.render(height=256, width=256, camera_id=0)
