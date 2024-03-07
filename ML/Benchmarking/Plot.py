# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os.path
import re
from typing import MutableSequence
from operator import iand
from functools import reduce
import glob
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker, dates, lines
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns

from minihydra import just_args, instantiate


def plot(path, plot_experiments=None, plot_agents=None, plot_suites=None, plot_tasks=None, steps=None,
         write_tabular=False, plot_train=False, title='UnifiedML', x_axis='Step', verbose=False):

    # Save path
    path = Path(path + f'/{"Train" if plot_train else "Eval"}')
    path.mkdir(parents=True, exist_ok=True)

    # Max number of steps to plot
    if steps is None:
        steps = np.inf

    # Make sure non empty and lists, and gather names
    empty = True
    specs = [plot_experiments, plot_agents, plot_suites, plot_tasks]
    plot_name = ''
    for i, spec in enumerate(specs):
        if spec is not None:
            empty = False
            if not isinstance(spec, MutableSequence):
                specs[i] = [spec]
            # Plot name
            plot_name += "_".join(specs[i] if i == 0 or len(specs[i]) < 5 else (specs[i][:5] + ['etc'])) + '_'
    plot_name = plot_name.strip('.')
    if empty:
        return

    # Style

    # RdYlBu, Set1, Set2, Set3, gist_stern, icefire, tab10_r, Dark2
    possible_palettes = ['Accent', 'RdYlBu', 'Set1', 'Set2', 'Set3', 'gist_stern', 'icefire', 'tab10_r', 'Dark2']
    # Note: finite number of color palettes: could error out if try to plot a billion tasks in one figure
    palette_colors = sum([sns.color_palette(palette) for palette in possible_palettes], [])

    # Universal theme
    sns.set_theme(font_scale=0.7, rc={'legend.loc': 'lower right', 'figure.dpi': 400,
                                      'legend.fontsize': 5.5, 'legend.title_fontsize': 5.5,
                                      # 'axes.titlesize': 4, 'axes.labelsize': 4, 'font.size': 4,
                                      # 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                                      # 'figure.titlesize': 4
                                      })

    # RETRIEVE DATA FOR PLOTTING
    performance, predicted_vs_actual, predicted_probas, min_steps = get_data(specs, steps, plot_train, verbose)

    universal_hue_order, palette = [], {}

    x_axis = x_axis.capitalize()

    pd.options.mode.chained_assignment = None

    if len(performance) > 0:
        universal_hue_order, handles = np.sort(performance.Agent.unique()), {}
        palette = {agent: color for agent, color in zip(universal_hue_order, palette_colors[:len(universal_hue_order)])}

        # PLOTTING (tasks)

        def make(cell_data, ax, ax_title, cell_palettes, hue_names, **kwargs):
            suite = ax_title.split('(')[1].split(')')[0]

            # Pre-processing
            x = x_axis if x_axis in cell_data.columns else 'Step'
            if x == 'Time':
                cell_data['Time'] = pd.to_datetime(cell_data['Time'], unit='s')

            # y = 'Accuracy' if 'classify' in suite.lower() \
            #     else 'MSE' if 'regression' in suite.lower() else 'Reward' if 'reward' in cell_data.columns \
            #     else [col for col in cell_data.columns if col not in ['time', 'step', 'frame', 'epoch', 'fps']][0]
            y = [col for col in cell_data.columns if col.lower()
                 not in ['time', 'step', 'frame', 'epoch', 'fps', 'agent', 'suite', 'task', 'seed', 'episode']
                 and not cell_data[col].isnull().values.any()][0]

            sns.lineplot(x=x, y=y, data=cell_data, ci='sd', hue='Agent', hue_order=np.sort(hue_names), ax=ax,
                         palette=cell_palettes)

            # Post-processing
            if x == 'Time':
                ax.set_xlabel("Time (h)")
                ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # For now group x axis into bins only for time

            if 'classify' in suite.lower():
                ax.set_ybound(0, 1.01)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            # Legend in subplots
            ax.legend(frameon=False).set_title(None)

            # Rotate x-axis names
            ax.tick_params(axis='x', rotation=20)

        general_plot(performance, path, plot_name + 'Tasks.png', palette, make, 'Task', title)

        # PLOTTING (suites)

        def make(ax, cell_data, ax_title, cell_palettes, hue_names, **kwargs):
            # Pre-processing
            x = x_axis if x_axis in cell_data.columns else 'Step'
            if x == 'Time':
                cell_data['Time'] = pd.to_datetime(cell_data['Time'], unit='s')

            # y = 'Accuracy' if 'classify' in ax_title.lower() \
            #     else 'MSE' if 'regression' in ax_title.lower() else 'Reward' if 'reward' in cell_data.columns \
            #     else [col for col in cell_data.columns if col not in ['time', 'step', 'frame', 'epoch', 'fps']][0]
            y = [col for col in cell_data.columns if col.lower()
                 not in ['time', 'step', 'frame', 'epoch', 'fps', 'agent', 'suite', 'task', 'seed', 'episode']
                 and not cell_data[col].isnull().values.any()][0]

            # Normalize
            for task in cell_data.Task.unique():
                for t in low:
                    if t.lower() in task.lower():
                        cell_data.loc[cell_data['Task'] == task, y] -= low[t]
                        cell_data.loc[cell_data['Task'] == task, y] /= high[t] - low[t]
                        continue

            sns.lineplot(x=x, y=y, data=cell_data, ci='sd', hue='Agent', hue_order=np.sort(hue_names), ax=ax,
                         palette=cell_palettes)

            # Post-processing
            if x == 'Time':
                ax.set_xlabel("Time (h)")
                ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
                # For now, group x axis into bins only for time
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

            if ax_title.lower() == 'atari':
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Human-Normalized Score')
            elif ax_title.lower() == 'dmc':
                ax.set_ybound(0, 1001)
            elif ax_title.lower() == 'classify':
                ax.set_ybound(0, 1.01)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            # Legend in subplots
            ax.legend(frameon=False).set_title(None)

            # Rotate x-axis names
            ax.tick_params(axis='x', rotation=20)

        general_plot(performance, path, plot_name + 'Suites.png', palette, make, 'Suite', title)

        # WRITING (tabular)

        # Consistent steps across all tasks for bar plot and tabular data
        min_time = performance.loc[performance['Step'] == min_steps, x_axis].unique()  # Checks if specified x-axis "time" is shared
        if len(min_time) > 1:  # If not, just use Step
            x_axis = 'Step'
            min_time = min_steps
        else:
            min_time = min_time[0]  # if so, use that time

        # Only show results for a consistent step count
        # TODO Allow results for differing step counts (non-uniform step count).
        #  Otherwise, RL and Classify become hard to plot together since RL logs by the 1000s while Classify
        #  logs by the 100s
        performance = performance[performance['Step'] == min_steps]

        # Score name y-axis
        metrics = [metric
                   for metric in performance.columns if metric.lower() not in
                   ['time', 'step', 'frame', 'epoch', 'fps', 'agent', 'suite', 'task', 'seed', 'episode']]

        # Use Reward or Accuracy as "Score"
        performance['Score'] = performance[metrics[0]]

        if len(metrics) > 1:
            # Where N/A, use the other
            performance.loc[performance['Score'].isna(), 'Score'] = performance[metrics[1]]

        # Mean and Median scores across seeds
        performance = performance[['Task', 'Suite', 'Agent', 'Score']].groupby(
            ['Task', 'Suite', 'Agent']).agg(Mean=('Score', np.mean), Median=('Score', np.median)).reset_index()

        # Normalized Mean and Median scores across seeds
        for i, (task_suite, suite) in performance[['Task', 'Suite']].iterrows():
            performance.loc[i, 'Normalized Mean'] = performance.loc[i, 'Mean']
            performance.loc[i, 'Normalized Median'] = performance.loc[i, 'Median']

            # Index norm lows/highs by task name or suite name
            task = task_suite.split(' (')[0]
            name = task if task in low and task in high else suite

            if name in low and name in high:
                # Mean
                performance.loc[i, f'Normalized Mean'] -= low[name]
                performance.loc[i, f'Normalized Mean'] /= high[name] - low[name]

                # Median
                performance.loc[i, f'Normalized Median'] -= low[name]
                performance.loc[i, f'Normalized Median'] /= high[name] - low[name]

        # Writes normalized metrics for this step out to csv
        if write_tabular:
            # Tasks Tabular
            performance.to_csv(path / (plot_name + f'{int(min_steps)}-Steps_Tasks_Tabular.csv'), index=False)

            # Suites Tabular - Mean or median of scores per suite
            metrics = {name: (seed_agg_name, agg) for seed_agg_name in ['Normalized Mean', 'Normalized Median']
                       for name, agg in [('Mean ' + seed_agg_name, np.mean), ('Median ' + seed_agg_name, np.median)]}
            performance[['Suite', 'Normalized Median', 'Normalized Mean']].groupby(['Suite']).agg(**metrics).reset_index(). \
                to_csv(path / (plot_name + f'{int(min_steps)}-Steps_Suites_Tabular.csv'), index=False)

        # PLOTTING (bar plot)

        def make(ax, cell_data, cell_palettes, hue_names, cell, **kwargs):
            # Pre-processing
            cell_data['Task'] = cell_data['Task'].str.split('(').str[0]

            sns.barplot(x='Task', y='Normalized Median', ci='sd', hue='Agent', data=cell_data, ax=ax,
                        hue_order=np.sort(hue_names), palette=cell_palettes)

            # Post-processing
            suite = cell[0]

            if x_axis.lower() == 'time':
                time_str = pd.to_datetime(min_time, unit='s').strftime('%H:%M:%S')
                ax.set_title(f'{suite} (@{time_str}h)')
            else:
                ax.set_title(f'{suite} (@{min_time:.0f} {x_axis}s)')

            if suite.lower() == 'atari':
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel('Median Human-Normalized')
            elif suite.lower() == 'dmc':
                ax.set_ybound(0, 1000)
                ax.set_ylabel('Median Reward')
            elif suite.lower() == 'classify':
                ax.set_ybound(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
                ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate('{:.0f}'.format(height) if suite.lower() not in ['atari', 'classify'] else f'{height:.0%}',
                            (x + width/2, y + height), ha='center', size=max(min(24 * width, 7), 5),  # No max(keep, 5)?
                            # color='#498057'
                            # color='#3b423d'
                            )

            # Rotate x-axis names and remove axis title
            ax.tick_params(axis='x', rotation=20)
            ax.set(xlabel=None)

            # Adaptive resizing
            # for patch in ax.patches:
            #     diff = patch.get_height() - new_value
            #
            #     # Set bar width
            #     patch.set_height(new_value)
            #
            #     # Recenter bar
            #     patch.set_y(patch.get_y() + diff * .5)

        # Max Agents for a Task - for configuring Bar Plot width
        mean_num_bars_per_task = performance.groupby(['Task', 'Agent']).size().reset_index().groupby(['Task']).size().mean()
        # max_num_bars_per_task = performance.groupby(['Task', 'Agent']).size().reset_index().groupby(['Task']).size().max()
        num_tasks = len(performance.Task.unique())
        # num_bars = len(performance.groupby(['Task', 'Agent']).size().reset_index().index)
        legend_width = max([len(name) for name in performance.Agent.unique()])

        general_plot(performance, path, plot_name + 'Bar.png', palette, make, 'Suite', title, 'Agent', True,
                     figsize=(max(4, num_tasks * mean_num_bars_per_task * 0.7) + legend_width / 30, 3))  # Adapt width

    if len(predicted_vs_actual) > 0:
        # Use previously defined palette if available, or add new colors from the universal hue order w/o conflict
        new_hues = 0
        for agent in np.sort(predicted_vs_actual.Agent.unique()):
            if agent not in palette:
                palette[agent] = palette_colors[len(universal_hue_order) + new_hues]
                new_hues += 1

        original_predicted_vs_actual = predicted_vs_actual.copy()

        # step = predicted_vs_actual[['Task', 'Step']].groupby('Task').max().reset_index() \
        #     if 'Step' in predicted_vs_actual.columns else None
        step = predicted_vs_actual[['Task', 'Step']].groupby('Task').max().reset_index()  # TODO Use this after XRD

        predicted_vs_actual['Accuracy'] = 0
        predicted_vs_actual.loc[predicted_vs_actual['Predicted'] == predicted_vs_actual['Actual'], 'Accuracy'] = 1
        predicted_vs_actual['Count'] = 1
        predicted_vs_actual.drop(['Predicted'], axis=1)
        predicted_vs_actual = predicted_vs_actual.rename(columns={'Actual': 'Class Label'})

        num_seeds = predicted_vs_actual.groupby(['Class Label', 'Agent', 'Task'])['Seed'].value_counts()
        num_seeds = num_seeds.groupby(['Class Label', 'Agent', 'Task']).count().reset_index()

        predicted_vs_actual = predicted_vs_actual.groupby(['Class Label', 'Agent', 'Task']).agg(
            {'Accuracy': 'sum', 'Count': 'size'}).reset_index()
        predicted_vs_actual['Accuracy'] /= predicted_vs_actual['Count']
        predicted_vs_actual['Count'] /= num_seeds['Seed']

        # Plotting (class sizes)

        def make(ax, ax_title, cell_data, cell, hue_names, cell_palettes, **kwargs):
            sns.scatterplot(data=cell_data, x='Class Label', y='Accuracy', hue='Agent', size='Count',
                            alpha=0.7, hue_order=np.sort(hue_names), ax=ax, palette=cell_palettes)

            #  Post-processing
            step_ = f' (@{int(step.loc[step["Task"] == cell[0], "Step"])} Steps)'
            # step_ = ' (@500000 Steps)'  # TODO This is just for XRD
            ax.set_title(f'{ax_title}{step_}')
            ax.set_ybound(-0.05, 1.05)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel(f'{"Train" if plot_train else "Eval"} Accuracy')

        general_plot(predicted_vs_actual, path, plot_name + 'ClassSizes.png', palette,
                     make, 'Task', title, 'Agent', True, False, False)

        # Plotting (heatmap)

        def make(cell_data, ax, cell_palettes, hue_names, **kwargs):
            # Pre-processing
            cell_data = pd.crosstab(cell_data.Predicted, cell_data.Actual)  # To matrix
            cell_data = cell_data.div(cell_data.sum(axis=0), axis=1)  # Normalize

            sns.heatmap(cell_data, ax=ax, linewidths=.5, vmin=0, vmax=1,  # Normalizes color bar in [0, 1]
                        cmap=sns.light_palette(cell_palettes[hue_names[0]], as_cmap=True))

            # Post-processing
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
            # ax.invert_yaxis()  # Can optionally represent vertically flipped

        general_plot(original_predicted_vs_actual, path, plot_name + 'Heatmap.png', palette,
                     make, ['Task', 'Agent'], title, 'Agent', False, True)

        if len(predicted_probas) > 0:
            pass


def get_data(specs, steps=np.inf, plot_train=False, verbose=False):
    # All CSVs from path, recursive
    csv_names = glob.glob('./Benchmarking/*/*/*/*/*.csv', recursive=True)

    performance = []
    predicted_vs_actual = []
    predicted_probas = []

    min_steps = steps

    # Parsing + reading
    for csv_name in csv_names:
        # Parse file names
        experiment, agent, suite, task_seed_eval = csv_name.split('/')[3:]
        split_size = 3 if 'Generate' in task_seed_eval else 5 if 'Predicted_vs_Actual' in task_seed_eval \
            else 4 if 'Predicted_Probas' in task_seed_eval else 2
        task_seed = task_seed_eval.rsplit('_', split_size)
        task, seed, eval = task_seed[0], task_seed[1], '_'.join(task_seed[2:]).replace('.csv', '')

        # Map suite names to properly-cased names
        suite = {k.lower(): k for k in ['Atari', 'DMC', 'Classify']}.get(suite.lower(), suite)

        # Whether to include this CSV
        include = True

        mode = 'train' if plot_train else 'eval'

        if eval.lower() not in [mode, f'predicted_vs_actual_{mode}', f'predicted_probas_{mode}']:
            include = False

        # Include based on spec
        datums = [experiment, agent, suite.lower(), task]
        for i, spec in enumerate(specs):
            if spec is not None and not re.match('^(%s)+$' % '|'.join(spec).replace('(', r'\(').replace(
                    ')', r'\)').replace('+', r'\+'), datums[i], re.IGNORECASE):
                if i == 3 and re.match('^.*(%s)+$' % '|'.join(spec).replace('(', r'\(').replace(
                        ')', r'\)').replace('+', r'\+'), datums[i], re.IGNORECASE):  # Tasks can be specified with suite
                    break
                include = False
                break

        if not include:
            continue

        # Add CSV
        csv = pd.read_csv(csv_name)

        # Track max step total across all csvs
        if 'step' in csv.columns and 'predicted_vs_actual' not in eval.lower() \
                and 'predicted_probas' not in eval.lower():
            length = int(csv.loc[csv['step'] <= steps, 'step'].max())
            # if length == 0:
            #     continue

            # Min number of max steps over all tasks/experiments  TODO per task/suite or range
            # Assumes data available for a single shared step across tasks - may not be true when log steps inconsistent
            min_steps = min(min_steps, length)

            if verbose and length < steps != np.inf:
                print(f'[Experiment {experiment} Agent {agent} Suite {suite} Task {task} Seed {seed}] '
                      f'has {length} steps.')

        csv['Agent'] = agent + ' (' + experiment + ')'  # Name Agent together with experiment
        csv['Suite'] = suite
        csv['Task'] = task + ' (' + suite + ')'  # Name Task together with suite
        csv['Seed'] = seed

        # Rolling max per run (as in CURL, SUNRISE)
        # This was critiqued heavily in https://arxiv.org/pperformance/2108.13264.pperformance
        # max_csv = csv.copy()
        # max_csv['reward'] = max_csv[['reward', 'step']].rolling(length, min_periods=1, on='step').max()['reward']

        if 'predicted_vs_actual' in eval.lower():
            predicted_vs_actual.append(csv)
        elif 'predicted_probas' in eval.lower():
            predicted_probas.append(csv)
        else:
            performance.append(csv)

    if len(performance):
        # To csv
        performance = pd.concat(performance, ignore_index=True)

        # Capitalize column names (uppercase and assume acronym if length <= 3)
        performance.columns = [' '.join([name[0].capitalize() + name[1:]
                                         for name in re.split(r'_|\s+', col_name)]) if len(col_name) > 3
                               else col_name.upper() for col_name in performance.columns]

        # Steps cap
        if steps < np.inf:
            performance = performance[performance['Step'] <= steps]

    if len(predicted_vs_actual):
        # To csv
        predicted_vs_actual = pd.concat(predicted_vs_actual, ignore_index=True)

        # Capitalize column names
        predicted_vs_actual.columns = [' '.join([name[0].capitalize() + name[1:] for name in re.split(r'_|\s+',
                                                                                                      col_name)])
                                       for col_name in predicted_vs_actual.columns]

        # To int
        predicted_vs_actual = predicted_vs_actual.astype({'Predicted': int, 'Actual': int})
    if len(predicted_probas):
        # To csv
        predicted_probas = pd.concat(predicted_probas, ignore_index=True)

        # Capitalize column names
        predicted_probas.columns = [' '.join([name[0].capitalize() + name[1:] for name in re.split(r'_|\s+', col_name)])
                                    for col_name in predicted_probas.columns]

        # To softmax multiple columns

    return performance, predicted_vs_actual, predicted_probas, min_steps


def general_plot(data, path, plot_name, palette, make_func, per='Task', title='UnifiedML', hue='Agent',
                 legend_aside=False, universal_legend=False, legend_title=True, figsize=None):
    if not isinstance(per, list):
        per = [per]

    cells = data[per].groupby(per).size().reset_index().drop(columns=0)

    total = len(cells.index)

    # Manually compute a full grid of full square-ish shape
    extra = 0
    num_rows = int(np.floor(np.sqrt(total)))
    while total % num_rows != 0:
        num_rows -= 1
    num_cols = total // num_rows

    # If too rectangular, automatically infer num cols/rows as square with empty extra cells
    if num_cols / num_rows > 5:
        num_cols = int(np.ceil(np.sqrt(total)))
        num_rows = int(np.ceil(total / num_cols))
        extra = num_rows * num_cols - total

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 3 * num_rows) if figsize is None else figsize)

    # Title
    if title is not None:
        fig.suptitle(title, y=0.99 if total > 10 else None)  # y controls height of title
        # fig.suptitle(title)

    cell_palettes = {}

    for i, cell in cells.iterrows():
        cell_data = data[reduce(iand, [data[per_name] == cell_name for per_name, cell_name in zip(per, cell)])]

        # Unique colors for this cell
        hue_names = cell_data[hue].unique()

        # No need to show Agent name in legend if all same - TODO perhaps across the whole plot, not per cell
        if len((data if universal_legend else cell_data)[hue].str.split('(').str[0].unique()) == 1:  # Unique Agent name
            # Remove Agent name from data columns
            cell_data[hue] = cell_data[hue].str.split('(').str[1:].str.join('(').str.split(')'
                                                                                           ).str[:-1].str.join(')')
            # Remove Agent name from legend
            for j, hue_name in enumerate(hue_names):
                hue_names[j] = ')'.join('('.join(hue_name.split('(')[1:]).split(')')[:-1])
                cell_palettes.update({hue_names[j]: palette[hue_name]})

            # Remove Agent from cell names
            if hue in per:
                cell[per.index(hue)] = ')'.join('('.join(cell[per.index(hue)].split('(')[1:]).split(')')[:-1])
        else:
            cell_palettes.update({hue_name: palette[hue_name] for hue_name in hue_names})

        # # Underscores as spaces for data columns
        # cell_data[hue] = cell_data[hue].str.replace('_', ' ')
        # # Underscores as spaces for legend
        # for j, hue_name in enumerate(hue_names):
        #     hue_names[j] = hue_name.replace('_', ' ')
        #     cell_palettes.update({hue_names[j]: cell_palettes.pop(hue_name)})
        # # Underscores as spaces for cell names
        # if hue in per:
        #     cell[per.index(hue)] = cell[per.index(hue)].replace('_', ' ')

        # Rows and cols
        row = i // num_cols
        col = i % num_cols

        # Empty extras
        if row == num_rows - 1 and col > num_cols - 1 - extra:
            break

        # Cell plot ("ax")
        ax = axs[row, col] if num_rows > 1 and num_cols > 1 else axs[col] if num_cols > 1 \
            else axs[row] if num_rows > 1 else axs

        # Cell title
        ax_title = ' '.join([name[0].upper() + name[1:] for name in cell[0].split('_')])

        if len(per) > 1:
            ax_title += ' :  ' + ' :  '.join(cell[1:])

        ax.set_title(ax_title)

        make_func(**locals())

        # Hide legend title
        if not legend_title:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])

        # Legend next to subplots
        if legend_aside:
            ax.legend(**{} if legend_title else {'handles': handles[1:], 'labels': labels[1:]},
                      loc=2, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, frameon=False,)

    # Universal legend
    if universal_legend:
        # Color order in legend between hue <-> palette
        hue_order = np.sort([hue_name for hue_name in cell_palettes])

        ax = axs[0, -1] if num_rows > 1 and num_cols > 1 else axs[-1] if num_cols > 1 \
            else axs[0] if num_rows > 1 else axs
        handles = [lines.Line2D([0], [0], marker='o', color=cell_palettes[label], label=label, linewidth=0)
                   for label in hue_order]
        ax.legend(handles, hue_order, loc=2, bbox_to_anchor=(1.25, 1.05),
                  borderaxespad=0, frameon=False).set_title(hue)

    for i in range(extra):
        fig.delaxes(axs[num_rows - 1, num_cols - i - 1])

    plt.tight_layout()
    plt.savefig(path / plot_name)

    plt.close()


# Lows and highs for normalization

atari_random = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
}
atari_human = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}

low = {**atari_random}
high = {**atari_human}


def main():
    args = just_args()
    defaults = just_args(source=f'{os.path.dirname(__file__)}/../Hyperparams/args.yaml').plotting
    instantiate(plot, **{**defaults, **args})


if __name__ == "__main__":
    main()
