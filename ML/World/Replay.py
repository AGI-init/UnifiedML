# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import random
from pathlib import PosixPath
from threading import Thread, Lock
from math import inf
import os

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch import multiprocessing as mp

from World.Memory import Memory, Batch
from World.Dataset import load_dataset, datums_as_batch, get_dataset_path, worker_init_fn, compute_stats

from Utils import Modals

from minihydra import instantiate, open_yaml, Args


class Replay:
    def __init__(self, path='Replay/', batch_size=1, device='cpu', num_workers=0, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 save=False, load=False, mem_size=None,
                 partition_workers=False, with_replacement=False, done_episodes_only=None, shuffle=True,
                 fetch_per=None, prefetch_factor=3, pin_memory=False, pin_device_memory=False, rewrite_shape=None,
                 dataset=None, transform=None, index='step', frame_stack=1, nstep=None, discount=1, agent_specs=None):

        self.device = device
        self.offline = offline
        self.epoch = 1
        self.nstep = nstep or 0  # Future steps to compute cumulative reward from
        self.stream = stream

        self.begin_flag = Flag()  # Wait until first call to sample before initial fetch

        self.last_batch_size = None

        if self.stream:
            return

        self.trajectory_flag = Flag()  # Tell worker to include experience trajectories as well

        if self.offline:
            self.begin_flag.set()

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.memory = Memory(num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity,
                             index=index)

        self.rewrite_shape = rewrite_shape  # For rewritable memory

        self.add_lock = Lock() if hd_capacity == inf else None  # For adding to memory in concurrency

        if isinstance(path, PosixPath):
            path = path.as_posix()  # TODO Probably better to use PosixPath universally

        dataset_config = dataset or Args(_target_=None)
        dataset_config['capacities'] = sum(self.memory.capacities)
        card = Args({'_target_': dataset_config}) if isinstance(dataset_config, str) else dataset_config
        # Perhaps if Online, include whether discrete -> continuous, since action shape changes in just that case

        # Optional specs that can be set based on data
        norm, standardize, obs_spec, action_spec = [getattr(agent_specs, spec, None)
                                                    for spec in ['norm', 'standardize', 'obs_spec', 'action_spec']]

        if dataset_config is not None and dataset_config._target_ is not None:
            # TODO Can system-lock w.r.t. save path if load_dataset is Dataset and Offline, then recompute load_dataset
            #   Only one process should save a previously non-existent memory at a time

            if offline:
                root = 'World/ReplayBuffer/Offline/'  # TODO can't control root?
                save_path = root + get_dataset_path(dataset_config, root)
            else:
                save_path = 'World/ReplayBuffer/Online/' + path

            # Memory save-path
            self.memory.set_save_path(save_path)

            # Batch-wise data augmentation on Dataset prior to storage in accelerated Memory
            aug = Modals(instantiate(dataset_config.get('aug'), device=device))

            # Pytorch Dataset or Memory path
            dataset = load_dataset('World/ReplayBuffer/Offline/', dataset_config) if offline else save_path

            # Fill Memory
            if isinstance(dataset, str):
                # Load Memory from path
                if os.path.exists(dataset + 'card.yaml'):
                    self.memory.load(dataset, desc=f'Loading Replay from {dataset}')
                    card = open_yaml(dataset + 'card.yaml')  # TODO Doesn't exist for unsaved mmap'd Online?
            else:
                batches = DataLoader(dataset, batch_size=mem_size or batch_size)

                # Add Dataset into Memory in batch-size chunks
                capacity = sum(self.memory.capacities)
                with tqdm(total=len(batches), desc='Loading Dataset into accelerated Memory...') as bar:
                    for i, data in enumerate(batches):
                        if len(self.memory) + len(next(iter(data.values())) if isinstance(data, (Args, dict))
                                                  else data[-1]) > capacity:
                            bar.total = i
                            break

                        # TODO This should probably work on Memory loading as well or be called dataset.aug
                        if aug is not None:
                            data = aug(data)

                        self.memory.add(datums_as_batch(data, done=i == len(batches) - 1))
                        bar.update()
                    bar.refresh()

            if action_spec is not None and action_spec.discrete:
                if 'discrete_bins' not in action_spec or action_spec.discrete_bins is None:
                    action_spec['discrete_bins'] = len(card.classes)

                if 'high' not in action_spec or action_spec.high is None:
                    action_spec['high'] = len(card.classes) - 1

                if 'low' not in action_spec or action_spec.low is None:
                    action_spec['low'] = 0
        elif not offline:
            self.memory.set_save_path('World/ReplayBuffer/Online/' + path)

            # Load Memory from path
            dataset = 'World/ReplayBuffer/Online/' + path
            if os.path.exists(dataset + 'card.yaml'):
                if load:
                    self.memory.load(dataset, desc=f'Loading Replay from {dataset}')
                    card = open_yaml(dataset + 'card.yaml')
                else:
                    for f in tqdm(os.listdir(dataset), desc=f'Deleting pre-existing Replay from {dataset}'):
                        os.remove(os.path.join(dataset, f))

        card['capacities'] = sum(self.memory.capacities)

        # Save Online replay on terminate
        if not offline and save:
            self.memory.set_save_path('World/ReplayBuffer/Online/' + path)
            atexit.register(lambda: (self.memory.save(desc='Saving Replay Memory...', card=card),
                                     print('Successfully saved Replay Memory to', self.memory.save_path)))

        # TODO Add meta datum if meta_shape, and make sure add() also does - or make dynamic

        transform = Modals(instantiate(transform, memory=self.memory))

        # Sampler

        # This is crucial for Online RL for some reason. Defaults True for Online, otherwise defaults False Offline
        # Perhaps because in-progress episode is shorter, thus sampling it disproportionately and over-fitting

        # TODO support index == 'step'
        done_episodes_only = index == 'episode' and (done_episodes_only is None and not offline
                                                     or done_episodes_only or False)

        sampler = Sampler(data_source=self.memory,
                          shuffle=shuffle,
                          offline=offline,
                          with_replacement=with_replacement,
                          done_episodes_only=done_episodes_only)

        # Parallel worker for batch loading

        create_worker = Offline if offline else Online

        self.partitions = (num_workers if partition_workers else 1) + bool(done_episodes_only)  # To reproduce DrQV2

        fetch_per = 0 if offline else batch_size // num_workers if fetch_per is None else fetch_per

        worker = create_worker(memory=self.memory,
                               fetch_per=fetch_per,
                               sampler=None if offline else sampler,
                               partition_workers=partition_workers,
                               done_episodes_only=done_episodes_only,
                               begin_flag=self.begin_flag,
                               transform=transform,
                               index=index,
                               frame_stack=frame_stack or 1,
                               nstep=self.nstep,
                               trajectory_flag=self.trajectory_flag,
                               discount=discount)

        # Batch loading

        self.batches = torch.utils.data.DataLoader(dataset=worker,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory and 'cuda' in device,  # or pin_device_memory
                                                   # pin_memory_device=device if pin_device_memory else '',
                                                   prefetch_factor=prefetch_factor if num_workers else 2,
                                                   # shuffle=shuffle and offline,  # Not compatible with Sampler
                                                   sampler=sampler if offline else None,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=bool(num_workers))

        # TODO Note: if loading agent with stats saved, this is unnecessary/redundant/perhaps the wrong stats
        # Fill in necessary obs_spec and action_spec stats from dataset
        if offline:
            if norm and ('low' not in obs_spec or 'high' not in obs_spec
                         or obs_spec.low is None or obs_spec.high is None) \
                    or standardize and ('mean' not in obs_spec or 'stddev' not in obs_spec
                                        or obs_spec.mean is None or obs_spec.stddev is None):
                if 'stats' not in card:  # TODO Store back to card if card already exists!
                    card['stats'] = compute_stats(self.batches)  # TODO Lock this after checking if card exists in path
                if obs_spec is not None:
                    obs_spec.update(card.stats)

        # Save to hard disk if Offline  TODO Lock this after checking if card exists in path
        if isinstance(dataset, Dataset) and offline:
            # if accelerate and self.memory.num_batches <= sum(self.memory.capacities[:-1]):
            #     root = 'World/ReplayBuffer/Offline/'
            #     self.memory.set_save_path(root + get_dataset_path(dataset_config, root))
            #     save = True
            if save or self.memory.num_batches > sum(self.memory.capacities[:-1]):  # Until save-delete check
                self.memory.save(desc='Memory-mapping Dataset for training acceleration and future re-use. '
                                      'This only has to be done once', card=card)
                print('Successfully saved Replay Memory to', self.memory.save_path)

        # Replay

        self._replay = None

    # Allows iteration via "next" (e.g. batch = next(replay))
    def __next__(self):
        if not self.begin_flag:
            self.begin_flag.set()

        if self.stream:
            # Environment streaming
            sample = self.stream
        else:
            # Replay sampling
            try:
                sample = next(self.replay)
            except StopIteration:
                self.epoch += 1
                self._replay = None  # Reset iterator when depleted
                sample = next(self.replay)

        self.last_batch_size = len(sample['obs'])

        return Batch({key: torch.as_tensor(value).to(device=self.device, non_blocking=True)
                      for key, value in sample.items()})

    def sample(self):
        return next(self)

    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)  # Recreates the iterator when exhausted
        return self._replay

    def include_trajectories(self):
        self.trajectory_flag.set()

    def add(self, trace):
        if trace is None:
            trace = []
        elif isinstance(trace, (Args, dict)):
            trace = [trace]

        for batch in trace:
            if self.stream:
                self.stream = batch  # For streaming directly from Environment  TODO N-step in {0, 1}
            else:
                def add():
                    with self.add_lock:
                        self.memory.add(batch)  # Add to memory

                # Don't thread if hd_capacity < inf,
                # TODO Has to be Queued, otherwise no guarentee of sequential
                #  TODO fix asynchronous add-induced deletion conflicting with worker __getitem__ of deleted index
                # if self.add_lock is None:
                #     self.memory.add(batch)  # Add to memory
                # else:
                #     Thread(target=add).start()  # Threading
                #     TODO Does a Lock block its own process; should be fine since that's the purpose of Thread lock
                self.memory.add(batch)  # Add to memory

    def set_tape(self, shape):
        self.rewrite_shape = shape or [0]

    def writable_tape(self, batch, ind, step):
        assert isinstance(batch, (dict, Batch)), f'expected \'batch\' to be dict or Batch, got {type(batch)}.'
        self.memory.writable_tape(batch, ind, step)

    def __len__(self):
        # Infinite if stream, else num episodes in Memory
        return int(9e9) if self.stream else len(self.memory)


class Worker:
    def __init__(self, memory, fetch_per, sampler, partition_workers, done_episodes_only, begin_flag, transform, index,
                 frame_stack, nstep, trajectory_flag, discount):
        self.memory = memory
        self.fetch_per = fetch_per  # TODO Default: batch size // num workers
        self.partition_workers = partition_workers
        self.done_episodes_only = done_episodes_only
        self.begin_flag = begin_flag

        self.sampler = None if sampler is None else OnlineSampler(sampler)
        self.samples_since_last_fetch = fetch_per

        self.transform = transform

        self.index = index

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.trajectory_flag = trajectory_flag
        self.discount = discount

        self.initialized = False

        self.dtypes = {}

    @property
    def worker(self):
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0

    def sample(self, index=-1, dynamic=False):
        if not self.initialized:
            self.memory.set_worker(self.worker)
            self.initialized = True

        # Periodically update memory
        while self.fetch_per and self.samples_since_last_fetch >= self.fetch_per:
            self.memory.update()  # Can make Online only

            if len(self.memory) and self.begin_flag:
                self.samples_since_last_fetch = 0
                break

        self.samples_since_last_fetch += 1

        if self.sampler is not None:
            index = next(self.sampler)

        cap = len(self.memory) - 1 - self.done_episodes_only  # Max episode index

        # Sample index
        if index == -1 or index > cap:
            index = random.randint(0, cap)  # Random sample an episode

        # Each worker can round index to their nearest allocated reciprocal to reproduce DrQV2 divide
        if self.partition_workers:
            while index == 0 and self.worker != 0 or index != 0 and index % len(self.memory.queues) != self.worker:
                index = (index + 1) % (cap + 1)

        # Retrieve from Memory
        if self.index == 'step':
            experience = self.memory[index]
            episode = experience.episode
            step = experience.step
        else:
            episode = self.memory[index]
            step = 0

        # TODO Since not randomly sampling for 'step' need to check relative to step!
        if dynamic:
            nstep = bool(self.nstep)  # Allows dynamic nstep if necessary
        else:
            nstep = self.nstep  # But w/o step as input, models can't distinguish later episode steps

        if len(episode) - step < nstep + 1:  # Try to make sure at least one nstep is present if nstep
            return self.sample(dynamic=True)

        if self.index == 'episode':
            step = random.randint(0, len(episode) - 1 - nstep)  # Randomly sample experience in episode

        experience = Args(episode[step])

        # Frame stack / N-step
        experience = self.compute_RL(episode, experience, step)

        # Transform
        experience = self.transform(experience)

        # Add metadata
        # TODO Don't store these unless needed
        # experience['episode_index'] = index
        # experience['episode_step'] = step
        if 'step' in experience:
            experience.pop('step')
        if 'done' in experience:
            experience.pop('done')

        for key in experience:  # TODO Move this adaptively in try-catch to collate converting first to int32
            if getattr(experience[key], 'dtype', None) == torch.int64:
                # For some reason, casting to int32 can throw collate_fn errors  TODO Lots of things do
                # experience[key] = experience[key].to(torch.float32)  # Maybe b/c some ints aren't as_tensor'd
                experience[key] = torch.as_tensor(experience[key], dtype=torch.int32)  # Ints just generally tend to crash
            elif getattr(experience[key], 'dtype', None) == torch.float64:
                experience[key] = torch.as_tensor(experience[key], dtype=torch.float32)
            # experience[key] = torch.as_tensor(experience[key], dtype=torch.float32).clone()
            # TODO In collate fn, just have a default tensor memory block to map everything to,
            #  maybe converts int64 to int32

            # Enforce consistency - Atari for example can have inconsistent dtypes
            if hasattr(experience[key], 'dtype'):
                if key not in self.dtypes:
                    self.dtypes[key] = experience[key].dtype
                elif experience[key].dtype != self.dtypes[key]:
                    experience[key] = torch.as_tensor(experience[key], dtype=self.dtypes[key])

        return experience.to_dict()

    def compute_RL(self, episode, experience, step):
        # TODO Just apply nstep and frame stack as transforms nstep, frame_stack, transform

        # Frame stack
        def frame_stack(traj, key, idx):
            frames = traj[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):  # If not enough frames, reuse the first
                frames = traj[:1] + frames
            frames = torch.concat([torch.as_tensor(frame[key])
                                   for frame in frames]).reshape(frames[0][key].shape[0] * self.frame_stack,
                                                                 *frames[0][key].shape[1:])
            return frames

        # Present
        if self.frame_stack > 1:
            experience.obs = frame_stack(episode, 'obs', step)  # Need experience as own dict/Batch for this

        # Future
        if self.nstep:
            # Transition
            experience.action = episode[step + 1].action

            traj_r = torch.as_tensor([float(experience.reward)
                                      for experience in episode[step + 1:step + self.nstep + 1]])

            experience['next_obs'] = frame_stack(episode, 'obs', step + len(traj_r))

            # Trajectory TODO
            if self.trajectory_flag:
                experience['traj_r'] = traj_r
                traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
                                         for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
                traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
                if 'label' in experience:
                    traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(len(traj_r) + 1)
            experience.reward = np.dot(discounts[:-1], traj_r).astype('float32')
            # experience['discount'] = 0 if episode[step + len(traj_r)].done else discounts[-1].astype('float32')
            experience['discount'] = discounts[-1].astype('float32')  # TODO Use above
        else:
            experience['discount'] = 1  # TODO just add via collate, not here, or check if exists in Q_learning

        return experience

    def __len__(self):
        return len(self.memory)


class Offline(Worker, Dataset):
    def __getitem__(self, index):
        return self.sample(index)  # Retrieve a single experience by index


class Online(Worker, IterableDataset):
    def __iter__(self):
        while True:
            yield self.sample()  # Yields a single experience


# Quick parallel one-time flag
class Flag:
    def __init__(self):
        self.flag = torch.tensor(False, dtype=torch.bool).share_memory_()
        self._flag = False

    def set(self):
        self.flag[...] = self._flag = True

    def __bool__(self):
        if not self._flag:
            self._flag = bool(self.flag)
        return self._flag


# Allows Pytorch Dataset workers to read from a sampler non-redundantly in real-time
class OnlineSampler:
    def __init__(self, sampler):
        self.main_worker = os.getpid()

        self.index = torch.zeros([], dtype=torch.int64).share_memory_()  # Int64

        self.read_lock = mp.Lock()
        self.read_condition = torch.tensor(False, dtype=torch.bool).share_memory_()
        self.index_condition = torch.tensor(False, dtype=torch.bool).share_memory_()

        self.sampler = sampler
        self.iterator = None

        Thread(target=self.publish, daemon=True).start()

    # Sample index publisher
    def publish(self):
        assert os.getpid() == self.main_worker, 'Only main worker can feed sample indices.'

        while True:
            # Wait until read is called in a process
            if self.read_condition:
                if self.iterator is None:
                    self.iterator = iter(self.sampler)
                try:
                    self.index[...] = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.sampler)
                    self.index[...] = next(self.iterator)

                # Notify that index has been updated  Since Iterable Dataset sequential, currently not needed
                self.read_condition[...] = False
                self.index_condition[...] = True

    def __iter__(self):
        yield from self

    def __next__(self):
        with self.read_lock:
            # Notify that read has been called
            self.read_condition[...] = True

            # Wait until index has been updated
            while True:
                if self.index_condition:
                    index = int(self.index)
                    self.index_condition[...] = False
                    return index


# class NullSampler:
#     def __iter__(self):
#         yield -1
#
#     def __len__(self):
#         return 0


# Sampling w/o replacement of Offline or dynamically-growing Online distributions
class Sampler:
    def __init__(self, data_source, shuffle=True, offline=True, with_replacement=False, done_episodes_only=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.offline = offline
        self.with_replacement = with_replacement

        if not self.offline and not self.with_replacement:
            assert shuffle, 'Online Sampler doesn\'t support not shuffling, got shuffle=False.'

        # TODO Check last_episode.done and Episode should have done attr
        self.done_episodes_only = done_episodes_only  # Whether to sample in-progress episodes

        self.size = len(self.data_source) - self.done_episodes_only

        self.indices = []

    def __iter__(self):
        if self.with_replacement:
            yield -1
        elif self.offline:
            yield from torch.randperm(self.size).tolist() if self.shuffle else list(range(self.size))
        else:
            size = len(self) - self.done_episodes_only
            if size > 0:
                # TODO Not compatible with deletions
                if not len(self.indices):
                    self.indices = list(range(size))
                elif size > self.size:
                    self.indices.extend(list(range(self.size, size)))
                self.size = size
                sample = random.randint(0, len(self.indices) - 1)
                last = self.indices[-1]
                self.indices[-1] = self.indices[sample]
                self.indices[sample] = last
                yield self.indices.pop()  # Ordinarily removing an element is O(n). pop is O(1).
            else:
                yield -1

    def __len__(self):
        return self.size if self.offline else len(self.data_source)
