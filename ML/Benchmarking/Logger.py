# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import csv
import datetime
import inspect
import re
import time
from pathlib import Path
from termcolor import colored

import numpy as np
import pandas as pd

import torch

from minihydra import get_module, instantiate


def shorthand(log_name):
    return ''.join([s[0].upper() for s in re.split('_|[ ]', log_name)] if len(log_name) > 3 else log_name.upper())


def format(log, log_name):
    k = shorthand(log_name)

    if 'time' in log_name.lower():
        log = str(datetime.timedelta(seconds=int(log)))
        return f'{k}: {log}'
    elif float(log).is_integer():
        log = int(log)
        return f'{k}: {log}'
    else:
        return f'{k}: {log:.04f}'


class Logger:
    def __init__(self, task, seed, generate=False, path='.', aggregation='mean', log_actions=False, wandb=False,
                 model=None, witness=None):

        self.path = path
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.task = task
        self.seed = seed
        self.generate = generate

        self.name = None

        self.logs = {}

        # "Predicted vs. Actual" - logged only for classify for now
        self.predicted = {} if log_actions else None
        self.probas = {} if log_actions else None

        self.aggregation = aggregation  # mean, median, last, max, min, or sum
        self.default_aggregations = {'step': np.ma.max, 'frame': np.ma.max, 'episode': np.ma.max, 'epoch': np.ma.max,
                                     'time': np.ma.max, 'fps': np.ma.mean}

        self.wandb = 'uninitialized' if wandb \
            else None

        # Agent-specific properties & logging
        self.model = model
        if witness is not None:
            self.witness(witness)

    def log(self, log, dump=False, exp=None):
        if log:

            if self.name not in self.logs:
                self.logs[self.name] = {}

            logs = self.logs[self.name]

            for log_name, item in log.items():
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                logs[log_name] = logs[log_name] + [item] if log_name in logs else [item]

            if self.predicted is not None and exp is not None:
                for exp in exp:
                    if self.name not in self.predicted:
                        self.predicted[self.name] = {'Predicted': [], 'Actual': []}
                    action = exp.action.squeeze()
                    if exp.action.shape[1] > 1:  # Assumes classification
                        if self.name not in self.probas:
                            self.probas[self.name] = []
                        self.probas[self.name].append(action)
                        # Corner case when Eval batch size is 1, batch dim gets squeezed out
                        for value in self.probas[self.name]:
                            if len(value.shape) <= 1:
                                value.shape = (1, *value.shape)
                        action = exp.action.argmax(1).squeeze()
                    self.predicted[self.name]['Predicted'].append(action)
                    self.predicted[self.name]['Actual'].append(exp.label.squeeze())
                    # Corner case when Eval batch size is 1, batch dim gets squeezed out
                    for key, value in self.predicted[self.name].items():
                        value[-1].shape = value[-1].shape or (1,)

        if dump:
            self.dump_logs()

    def dump_logs(self):
        if self.name is None:
            # Iterate through all logs
            for name in self.logs:
                for log_name in self.logs[name]:
                    agg = self.aggregate(log_name)
                    self.logs[name][log_name] = agg(self.logs[name][log_name])
                self._dump_logs(self.logs[name])
                self.dump_actions(self.logs[name])
            self.logs = {}
        else:
            # Iterate through just the named log
            if self.name not in self.logs:
                return
            for log_name in self.logs[self.name]:
                agg = self.aggregate(log_name)
                self.logs[self.name][log_name] = agg(self.logs[self.name][log_name])
            self._dump_logs(self.logs[self.name])
            self.dump_actions(self.logs[self.name])
            self.logs[self.name] = {}
            del self.logs[self.name]

    def dump_actions(self, logs):
        if self.predicted is not None and self.name in self.predicted \
                and len(self.predicted[self.name]['Predicted']) > 0 \
                and len(self.predicted[self.name]['Actual']) > 0:
            assert 'step' in logs

            file_name = Path(self.path) / f'{self.task}_{self.seed}_Predicted_vs_Actual_{self.name}.csv'

            for key in self.predicted[self.name]:
                self.predicted[self.name][key] = np.concatenate(self.predicted[self.name][key])

            df = pd.DataFrame(self.predicted[self.name])
            df['Step'] = int(logs['step'])
            df.to_csv(file_name, index=False)

            self.predicted[self.name] = {'Predicted': [], 'Actual': []}

        if self.probas is not None and self.name in self.probas:
            file_name = Path(self.path) / f'{self.task}_{self.seed}_Predicted_Probas_{self.name}.csv'

            self.probas[self.name] = np.concatenate(self.probas[self.name])

            df = pd.DataFrame(self.probas[self.name], columns=list(range(self.probas[self.name][0].shape[-1])))
            df['Step'] = int(logs['step'])
            df.to_csv(file_name, index=False)

            self.probas[self.name] = []

    # Aggregate list of scalars or batched-values of arbitrary lengths
    def aggregate(self, log_name):
        def last(data):
            data = np.array(data).flat
            return data[len(data) - 1]

        agg = self.default_aggregations.get(log_name,
                                            np.ma.mean if self.aggregation == 'mean'
                                            else np.ma.median if self.aggregation == 'median'
                                            else last if self.aggregation == 'last'
                                            else np.ma.max if self.aggregation == 'max'
                                            else np.ma.min if self.aggregation == 'min'
                                            else np.ma.sum)

        def size_agnostic_agg(stats):
            if isinstance(stats[0], (list, tuple, set)):
                stats = sum(stats, [])

            stats = [(stat,) if np.isscalar(stat) else stat.flatten() for stat in stats]

            masked = np.ma.empty((len(stats), max(map(len, stats))))
            masked.mask = True
            for m, stat in zip(masked, stats):
                m[:len(stat)] = stat  # Each 1-D array of logs added can be of different length
            return agg(masked)

        return agg if agg == last else size_agnostic_agg

    def _dump_logs(self, logs):
        self.dump_to_console(logs)
        self.dump_to_csv(logs)
        if self.wandb is not None:
            self.log_wandb(logs)

    def dump_to_console(self, logs):
        name = colored(self.name,
                       'yellow' if self.name.lower() == 'train' else 'green' if self.name.lower() == 'eval' else None,
                       attrs=['dark'] if self.name.lower() == 'seed' else None)
        pieces = [f'| {name: <14}']
        for log_name, log in logs.items():
            pieces.append(format(log, log_name))
        print(' | '.join(pieces))

    def remove_old_entries(self, logs, file_name):
        rows = []
        with file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= logs['step']:
                    break
                rows.append(row)
        with file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=logs.keys(),
                                    extrasaction='ignore',
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dump_to_csv(self, logs):
        logs = dict(logs)

        assert 'step' in logs

        name = self.name
        if self.generate:
            name = 'Generate_' + name

        file_name = Path(self.path) / f'{self.task}_{self.seed}_{name}.csv'

        write_header = True
        if file_name.exists():
            write_header = False
            self.remove_old_entries(logs, file_name)

        file = file_name.open('a')
        writer = csv.DictWriter(file,
                                fieldnames=logs.keys(),
                                restval=0.0)
        if write_header:
            writer.writeheader()

        writer.writerow(logs)
        file.flush()

    def log_wandb(self, logs):
        if self.wandb == 'uninitialized':
            import wandb

            experiment, agent, suite = self.path.split('/')[2:5]

            if self.generate:
                agent = 'Generate_' + agent

            wandb.init(project=experiment, name=f'{agent}_{suite}_{self.task}_{self.seed}', dir=self.path)

            for file in ['', '*/', '*/*/', '*/*/*/']:
                try:
                    wandb.save(f'./Hyperparams/{file}*.yaml')
                except Exception:
                    pass

            self.wandb = wandb

        measure = 'reward' if 'reward' in logs else 'accuracy'
        if measure in logs:
            logs[f'{measure} ({self.name})'] = logs.pop(f'{measure}')

        self.wandb.log(logs, step=int(logs['step']))

    # Add counter properties to agent & model scope (e.g. step, frame, episode, etc.)
    def witness(self, agent):
        # Set agent properties
        defaults = {'birthday': getattr(agent, 'birthday', time.time()),
                    'step': getattr(agent, 'step', 0),
                    'frame': getattr(agent, 'frame', 0),
                    'episode': getattr(agent, 'episode', 1),
                    'epoch': getattr(agent, 'epoch', 1)}

        setattr(agent, '_defaults_', defaults)

        targets = []

        if hasattr(agent, 'encoder') and hasattr(agent.encoder, '_eyes'):
            setattr(agent.encoder._eyes, '_defaults_', defaults)    # TODO Parallel
            targets.append(type(agent.encoder._eyes))
        if hasattr(agent, 'actor') and hasattr(agent.actor, '_pi_head'):
            setattr(agent.actor._pi_head.ensemble[0], '_defaults_', defaults)  # TODO Parallel
            targets.append(type(agent.actor._pi_head.ensemble[0]))

        for key, value in defaults.items():
            setattr(type(agent), key, property(lambda a, _key_=key: a._defaults_[_key_],
                                               lambda a, new_value, _key_=key: a._defaults_.update({_key_: new_value})))
            for target in targets:
                setattr(target, key, property(lambda m, _key_=key: m._defaults_[_key_],
                                              lambda m, new_value, _key_=key: m._defaults_.update({_key_: new_value})))

    def re_witness(self, logs, agent, replay):
        if logs is not None:
            logs.update(time=time.time() - agent.birthday, step=agent.step, frame=agent.frame, episode=agent.episode)

            # Online -> Offline conversion
            if replay.offline:
                agent.step += 1
                agent.frame += replay.last_batch_size
                agent.epoch = logs['epoch'] = replay.epoch
                logs['frame'] += 1  # Offline is 1 behind Online in training loop
                logs.pop('episode')
            else:
                logs.pop('epoch')

            self.log(logs)

    def seed(self):
        return self.mode('Seed')

    def train(self):
        return self.mode('Train')

    def eval(self):
        return self.mode('Eval')

    def mode(self, name):
        self.name = name
        return self
