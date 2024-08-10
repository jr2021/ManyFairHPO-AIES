'''
 This file has all methods to fetch data from experiment.pickle, fetch, extract metrics and plot, hypervolume curve,
 pareto fronts including aggregated plots
'''

import glob
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ax import Experiment, MultiObjectiveOptimizationConfig
from ax.core.base_trial import TrialStatus
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from MOHPOBenchExperimentUtils.utils.experiment_utils import load_experiment

plt.style.use('ggplot')

from matplotlib.cm import ScalarMappable


def _hypervolume_evolution_single(experiment: Experiment, surrogate=False):
    assert isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig), \
        'experiment must have an optimization_config of type MultiObjectiveOptimizationConfig '

    th_list = experiment.optimization_config.objective_thresholds
    metrics = [th.metric for th in th_list]

    ind_completed_runs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
    data = experiment.fetch_trials_data(trial_indices=ind_completed_runs).df

    if surrogate:
        # fetch training and prediction cost to add in wallclocktime for surrogate models
        surrogate_cost = data[(data['metric_name'] == 'timetrain') | (data['metric_name'] == 'cost')]['mean'].values
        surrogate_cost = np.cumsum(surrogate_cost)
        logger.debug(f'surrogate cost:{surrogate_cost}')

    data = [(-1 if m.lower_is_better else 1) * data[data['metric_name'] == m.name]['mean'].values for m in metrics]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.ascontiguousarray(np.asarray(data).T), device=device).float()

    ref_points = [(-1 if th.metric.lower_is_better else 1) * float(th.bound) for th in th_list]
    logger.debug("objective ref point:{}", ref_points)
    hv = Hypervolume(torch.tensor(ref_points, device=device))

    result = {
        'hypervolume': [0.0],
        'walltime': [0.0],
        'evaluations': [0.0]
    }

    # TODO: Adapt HV Volume Plots
    # Currently, all runs are used also all fidelities. However for MF methods this might be misleading.
    # Implement two versions: a) only highest fidelity
    #                         b) Similar to the trajectory computation in BOHB. use only the highest so far seen budget.
    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        if surrogate:
            #Add train and predict time
            total_runtime = ((experiment.trials[i - 1].time_completed -
              experiment.time_created).total_seconds()) + surrogate_cost[i-1]
            result['walltime'].append(total_runtime)
        else:
            result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))

        result['evaluations'].append(i)

    return pd.DataFrame(result)


def get_metrices(experiment, path: Union[str, Path]):
    path = Path(path)

    metrics = experiment.optimization_config.objective.metrics
    metrics = [obj.name for obj in metrics]
    th_list = experiment.optimization_config.objective_thresholds
    logger.debug("th_list:{}", th_list)

    ind_completed_runs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
    data = experiment.fetch_trials_data(trial_indices=ind_completed_runs).df

    values = [data[data['metric_name'] == obj]['mean'].values for obj in metrics]
    values = np.asarray(np.asarray(values)).T

    with open((path / f'all_metric_{experiment.name}.txt'), 'w') as f:
        np.savetxt(f, values)
    # with open(os.path.join(path, "metric_{}.txt".format(experiment.name)), 'w') as f:
    #     np.savetxt(f, values)

    maximizator = [(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list]
    values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
    values = values[values[:, 0].argsort()]

    with open((path / f"sortmetric_{experiment.name}.txt"), 'w') as f:
        np.savetxt(f, values)


def plot_hypervolume_evolution(
        experiments: List[Experiment],
):
    fig, axes = plt.subplots(4, figsize=(5, 5 * 4))

    for exp in tqdm(experiments, desc="Hypervolume Evolution"):
        data = _hypervolume_evolution_single(exp)

        plot = axes[0].plot(data['walltime'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[0].set_xlabel('Walltime')
        axes[0].set_ylabel('Hypervolume')
        axes[0].legend()
        axes[0].set_xscale('log')

        if 'SHEMOA' in exp.name:
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EHVI':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EASH':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'MOBOHB':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'BNC':
            axes[0].axvline(int(86400 * .35), color=plot[-1].get_color())
            axes[0].axvline(int(86400 * .75), color=plot[-1].get_color())

        axes[1].plot(data['evaluations'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[1].set_xlabel('Number of evaluations')
        axes[1].set_ylabel('Hypervolume')
        axes[1].legend()
        axes[1].set_xscale('log')

        if exp.name == 'MDEHVI':
            axes[1].axvline(50, color=plot[-1].get_color())
        if exp.name == 'BNC':
            axes[1].axvline(np.argmax(data['walltime'] > 86400 * .35), color=plot[-1].get_color())
            axes[1].axvline(np.argmax(data['walltime'] > 86400 * .75), color=plot[-1].get_color())

        ###############################
        plot = axes[2].plot(data['walltime'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[2].set_xlabel('Walltime')
        axes[2].set_ylabel('Hypervolume')
        axes[2].legend()

        if 'SHEMOA' in exp.name:
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EHVI':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EASH':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'MOBOHB':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'BNC':
            axes[2].axvline(int(86400 * .35), color=plot[-1].get_color())
            axes[2].axvline(int(86400 * .75), color=plot[-1].get_color())

        axes[3].plot(data['evaluations'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[3].set_xlabel('Number of evaluations')
        axes[3].set_ylabel('Hypervolume')
        axes[3].legend()

        if exp.name == 'MDEHVI':
            axes[3].axvline(50, color=plot[-1].get_color())
        if 'BNC' in exp.name:
            axes[3].axvline(np.argmax(data['walltime'] > 86400 * .35), color=plot[-1].get_color())
            axes[3].axvline(np.argmax(data['walltime'] > 86400 * .75), color=plot[-1].get_color())
    plt.savefig('hypervolume.pdf', dpi=450)


def plot_aggregated_hypervolume_time(experiments: Dict, path: str, log=False, valid_experiments=[]):
    experiments_by_type = defaultdict(list)
    for exp, val in experiments.items():
        experiments_by_type[exp.split('_')[0]].append(deepcopy(val))

    fig, axes = plt.subplots(1, figsize=(10, 5))

    for exp_type, values in experiments_by_type.items():

        if exp_type not in valid_experiments:
            continue

        if exp_type != 'BNC':
            timechange = np.mean([values[i]['walltime'][{
                'MDEHVI': 10,
                'MOBOHB': 10,
                'EASH': 25,
                'RS': 0,
                'MOSHBANANAS': 20,
            }[exp_type]] for i in range(len(values))]) / 3600
        else:
            # timechange = 23 * 3600 * (.35)
            timechange = 23 * (.35)
        df = None
        for i in range(len(values)):
            values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
            values[i] = values[i].resample('1h', on='walltime').max()[['hypervolume']]
            df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')

        df.columns = [f'hypervolume_{i}' for i in range(len(values))]
        df = df.pad()

        walltime = df.index.astype('timedelta64[h]').values
        mean = df.mean(axis=1).values
        std = df.std(axis=1).values
        axes.set_xlim(0, 25)
        plot = axes.plot(walltime, mean, '-', lw=3.0, label=exp_type)
        plt.axvline(timechange, color=plot[-1].get_color())
        # axes.fill_between(walltime, mean + 2 * std, mean - 2 * std, alpha=0.5)

    axes.set_xlabel('Walltime')
    axes.set_ylabel('Hypervolume')
    axes.legend()
    if log:
        axes.set_xscale('log')

    # plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.pdf', dpi=450)
    plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.jpg', dpi=450)


def plot_aggregated_hypervolume_evaluations(experiments: Dict, path: str, log=False, valid_experiments=[]):
    experiments_by_type = defaultdict(list)
    for exp, val in experiments.items():
        experiments_by_type[exp.split('_')[0]].append(val)

    fig, axes = plt.subplots(1, figsize=(10, 5))

    for exp_type, values in experiments_by_type.items():

        if exp_type not in valid_experiments:
            continue

        df = None
        for i in range(len(values)):
            values[i] = values[i].set_index('evaluations')[['hypervolume']]
            df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='evaluations')

        df.columns = [f'hypervolume_{i}' for i in range(len(values))]
        df = df.pad()

        print(exp_type)
        print(df.values[:4])
        print(np.std(df.values[:4], axis=1))

        evaluations = df.index.values
        mean = df.mean(axis=1).values
        std = df.std(axis=1).values
        plot = axes.plot(evaluations, mean, '-', lw=3.0, label=exp_type)
        plt.axvline(evaluations[{
            'MDEHVI': 10,
            'BNC': 0,
            'MOBOHB': 10,
            'EASH': 25,
            'RS': 0,
            'MOSHBANANAS': 20,
        }[exp_type]], color=plot[-1].get_color())
        # axes.fill_between(evaluations, mean + 2 * std, mean - 2 * std, alpha=0.5)

    axes.set_xlabel('Evaluations')
    axes.set_ylabel('Hypervolume')
    axes.legend()
    if log:
        axes.set_xscale('log')

    # plt.savefig(f'{path}/aggregated_evaluations_log_{str(log).lower()}.pdf', dpi=450)
    plt.savefig(f'{path}/aggregated_evaluations_log_{str(log).lower()}.jpg', dpi=450)


def plot_aggregated_scatter(
        path: str,
        experiments_names: List[str],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):
    experiments = [
        [
            load_experiment(e) for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
        ] for exp_name in experiments_names
    ]

    metrics = experiments[0][0].optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), len(experiments)))

    for ax, experiment_list in zip(axes, experiments):

        for i, experiment in enumerate(experiment_list[:5]):

            ind_completed_runs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
            data = experiment.fetch_trials_data(trial_indices=ind_completed_runs).df

            values_x = data[data['metric_name'] == metric_x.name]['mean'].values
            values_y = data[data['metric_name'] == metric_y.name]['mean'].values

            values_x = values_x if metric_x.lower_is_better else -1 * values_x
            values_y = values_y if metric_y.lower_is_better else -1 * values_y

            num_elements = values_x.size

            iterations = np.arange(num_elements)

            if i == 0:
                ax.set_title(experiment.name.split('_')[0])
                ax.set_xlabel(metric_x.name)
                ax.set_ylabel(metric_y.name)

            ax.scatter(values_x, values_y, c=iterations, alpha=0.8)

            th_list = experiment.optimization_config.objective_thresholds
            min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
            min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

            ax.set_xlim(right=8)
            if bound_max_x or True:
                ax.set_xlim(left=2)

            ax.set_ylim(top=0)
            if bound_max_x or True:
                ax.set_ylim(bottom=-110)

            if i == 0:
                norm = plt.Normalize(iterations.min(), iterations.max())
                sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.15)
                fig.colorbar(sm, cax=cax, label='Iterations')

    plt.savefig(f'{path}/evals_overtime.pdf', dpi=450)


def plot_aggregated_pareto_fronts(
        path: str,
        experiments_names: List[str],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):
    experiments = [
        [
            e for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
        ] for exp_name in experiments_names
    ]

    sample_experiment = load_experiment(experiments[0][0])
    metrics = sample_experiment.optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, ax = plt.subplots()

    th_list = sample_experiment.optimization_config.objective_thresholds
    # hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list][::-1], device=torch.device("cuda:0")))
    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list][::-1],
                                  device=torch.device("cpu")))
    for exp in experiments:
        if len(exp) == 0:
            continue

        print('Doing: ', exp)

        exp = load_experiment(exp[0])

        ind_completed_runs = exp.trial_indices_by_status[TrialStatus.COMPLETED]
        data = exp.fetch_trials_data(trial_indices=ind_completed_runs).df
        values_x = data[data['metric_name'] == metric_x.name]['mean'].values
        values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        maximizator = [-1 if metric_x.lower_is_better else 1, -1 if metric_y.lower_is_better else 1]

        values = np.asarray([values_x, values_y]).T
        values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
        values = values[values[:, 0].argsort()]

        # total_hv = int(hv.compute(torch.tensor(values * maximizator, device=torch.device("cuda:0"))))
        total_hv = int(hv.compute(torch.tensor(values * maximizator, device=torch.device("cpu"))))
        ax.plot(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp.name + ' ' + str(total_hv))

    ax.set_xlabel(metric_x.name)
    ax.set_ylabel(metric_y.name)

    th_list = sample_experiment.optimization_config.objective_thresholds
    min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
    min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

    ax.set_xlim(right=min_x)
    if bound_max_x:
        ax.set_xlim(left=bound_max_x)

    ax.set_ylim(top=min_y)
    if bound_max_x:
        ax.set_ylim(bottom=bound_max_y)

    ax.legend()

    plt.savefig(f'{path}/paretofronts.pdf', dpi=450)


def plot_pareto_fronts(
        experiments: List[Experiment],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):
    metrics = experiments[0].optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, ax = plt.subplots()

    th_list = experiments[0].optimization_config.objective_thresholds
    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list],
                                  device=torch.device("cuda:0")))

    for exp in experiments:
        ind_completed_runs = exp.trial_indices_by_status[TrialStatus.COMPLETED]
        data = exp.fetch_trials_data(trial_indices=ind_completed_runs).df
        values_x = data[data['metric_name'] == metric_x.name]['mean'].values
        values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        maximizator = [-1 if metric_x.lower_is_better else 1, -1 if metric_y.lower_is_better else 1]

        values = np.asarray([values_x, values_y]).T
        values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
        values = values[values[:, 0].argsort()]

        total_hv = hv.compute(torch.tensor(values * maximizator, device=torch.device("cuda:0")))

        ax.plot(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp.name + ' ' + str(total_hv))

    ax.set_xlabel(metric_x.name)
    ax.set_ylabel(metric_y.name)

    th_list = experiments[0].optimization_config.objective_thresholds
    min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
    min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

    ax.set_xlim(right=min_x)
    if bound_max_x:
        ax.set_xlim(left=bound_max_x)

    ax.set_ylim(top=min_y)
    if bound_max_x:
        ax.set_ylim(bottom=bound_max_y)

    ax.legend()

    plt.savefig('figures/paretofronts2.pdf', dpi=450)
