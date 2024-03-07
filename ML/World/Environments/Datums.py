# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
import os
import resource
import warnings

import numpy as np
import torch

from torch.utils.data import DataLoader

from World.Dataset import load_dataset, worker_init_fn, compute_stats
from minihydra import Args


class Datums:
    """
    Envs must:
    (1) have a step(action) function
    (2) have a reset() function

        The step() and reset() functions output experiences ("exp")
            i.e. dicts containing datums like "obs", "action", "reward", 'label", "done"

        In the step() function, instead of outputting one experience, can output a (prev, now) pair of experiences.
        Some datums, such as reward, are time-delayed and specifying them in a separate "prev" experience makes
        it possible to pair corresponding pairs without time-delay for metrics and Replay.

        The end of an episode or epoch may be marked by the "done" boolean. If unspecified, it will be assumed True
        during step() calls.
            - This sequence demarcation can be useful for aggregating metrics (e.g. based on multiple steps or an epoch
              of batches) or for the automatic organizing of temporal data in Replay.

            - The "done" step doesn't get acted on. Its contents can be a no-op (either output None or just
              {'done': True}), or, if you're using (prev, now) pairs, the prev experience in a done state should still
              include any additional datums needed for logging or storing in Replay, such as "reward". It shouldn't
              include the "done" key. The now state, which should hold the "done" key or be None, is ignored if it holds
              no other key, otherwise it's also sent to Replay and can be retrieved as a "next_obs" for RL for example.
            - The reset() state can't ever be a "done" state even if you specify it as such. In that case,
              output None from your step() function.

    Envs can:
    (1) have an obs_spec dict as an attribute
    (2) have an action_spec dict as an attribute
    (3) include a render() method, frame_stack(obs) method, and/or an action_repeat init arg that Env should adapt to
        and include as a self.action_repeat int attribute

        (1) and (2) if you want custom input/output spec info to be passed to your Agent/Model (as dicts/Args).
        The Environment infers many input/output specs by default. For example, obs_spec.shape often is automatically
        included since it can be inferred from an "obs"-key value of the reset() function. action_spec.shape
        from a "label" if present.

        - For obs_spec, besides "shape", these stats can include for example: "shape", "mean", "stddev", "low", "high".
        Useful for standardization and normalization.

        - For action_spec, these stats can include for example: "shape", "discrete_bins", "low", "high", "discrete".
        For discrete action spaces and action normalization. Many of these can be inferred,
        and see Environment.py "action_spec" @property for what the defaults are if left unspecified.

        All of this is optional and doesn't have to be used / can be ignored. Include obs_spec and action_spec as
        args in your Agent/Model's __init__ method to get access to the configured specs of the Env.

        See DMC.py and Atari.py for examples of frame_stack(obs) and action_repeat. Used commonly in RL.
        Video logging online rollouts can be facilitated by the render() method. Also can be ignored.

    ---

    Datasets must:
    (1) extend Pytorch Datasets
    (2) output (obs, label) pairs, or dicts of named datums, e.g., {'obs': obs, 'label': label, ...}

        Accessed in the API via dataset= and/or test_dataset=.

    Datasets can:
    (1) include a "classes" attribute that lists the different class names or classes

        This allows quick and exact computation of number of classes being predicted from, without having to count
        via iterating through the whole dataset.
    """
    def __init__(self, dataset, test_dataset=None, train=True, stream=False, batch_size=8, num_workers=1,
                 standardize=False, norm=False, device='cpu'):
        if not train and test_dataset is not None:
            # Assume test_dataset
            dataset = test_dataset

        # (This might not generally be necessary)
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)  # Shared memory can create a lot of file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))  # Increase soft limit to hard limit

        dataset = load_dataset('World/ReplayBuffer/Offline/', dataset, allow_memory=False, train=train)

        discrete = hasattr(dataset, 'classes')  # load_dataset adds a .classes attribute to the dataset if discrete

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.batches = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=torch.device(device).type == 'cuda',
                                  collate_fn=getattr(dataset, 'collate_fn', None),  # Useful if streaming dynamic lens
                                  worker_init_fn=worker_init_fn)  # TODO Support using Replay in Env (and by default)

        self._batches = iter(self.batches)

        self.num_sampled_batches = 0

        # Experience
        self.exp = None

        # Fill in necessary obs_spec and action_spec stats (e.g. mean, stddev, low, high) from dataset
        if (not train and stream) and (standardize or norm):
            # TODO load_dataset can recall stats from card as well; maybe allow specifying what stats to compute
            self.obs_spec = compute_stats(self.batches)  # TODO Should provide per-Datum specs

        if discrete:
            self.action_spec = {'discrete_bins': len(dataset.classes)}

    def step(self, action):
        if self.num_sampled_batches < len(self.batches):  # Episode is "done" at end of epoch
            return self.reset()  # Re-sample datums

    def reset(self):
        # Sample batch
        batch = self.sample()
        self.exp = Args(done=False, **batch)
        self.num_sampled_batches += 1

        # Convert sample to numpy  TODO Maybe not necessary
        for key, value in self.exp.items():
            self.exp[key] = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.array(value)

        # Label should have shape (batch_size, 1) if its shape is (batch_size,)
        if 'label' in self.exp and len(self.exp.label.shape) == 1:
            self.exp.label = np.expand_dims(self.exp.label, 1)

        # Return batch
        return self.exp

    def sample(self):
        try:
            return next(self._batches)
        except StopIteration:
            self.num_sampled_batches = 0
            self._batches = iter(self.batches)
            return next(self._batches)

    def render(self):
        # Assumes image dataset
        image = next(iter(self.sample().values())) if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image) - 1)]).transpose(1, 2, 0)  # Channels-last


# Mean of empty reward should be NaN, catch acceptable usage warning  TODO Delete if these warnings don't pop up anyway
warnings.filterwarnings("ignore", message='.*invalid value encountered in scalar divide')
warnings.filterwarnings("ignore", message='invalid value encountered in double_scalars')
warnings.filterwarnings("ignore", message='Mean of empty slice')
