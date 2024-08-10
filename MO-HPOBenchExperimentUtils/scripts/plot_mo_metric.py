import torch
import os
import argparse
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

import MOHPOBenchExperimentUtils.plotting
from MOHPOBenchExperimentUtils.plotting.plotting_utils import export_legend

from MOHPOBenchExperimentUtils.utils.settings_utils import load_settings, load_optimizer_settings
from scripts.check_and_create_missing_files import modified_check
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from botorch.utils.multi_objective.pareto import is_non_dominated

plt.style.use('ggplot')


# To mask some configurations randomly for better visualization of aggregated plots
def make_mask(data_size, sample_size):
    sample_size = int(sample_size)
    mask = np.array([True] * sample_size + [False] * (data_size - sample_size))
    np.random.shuffle(mask)
    return mask


def plot_aggregated_scatter(
        data,
        axes, fig,
        x_label,
        y_label,
        title,
        idx=0,
        last_idx=0,
        iterations_bar=False
):
    values_x = data[:, 0]
    values_y = data[:, 1]
    num_elements = values_x.size
    iterations = np.arange(num_elements)
    axes.set_xlabel(x_label, fontweight='bold')
    if idx == 0:
        axes.set_ylabel(y_label, fontweight='bold')
    axes.set_title(title, fontweight='bold')

    axes.xaxis.label.set_color('black')
    axes.yaxis.label.set_color('black')

    mask = make_mask(len(values_x), 0.1 * (len(values_x)))
    if iterations_bar:
        axes.scatter(values_x[mask], values_y[mask], c=iterations[mask], alpha=0.8)
        iterations = np.arange(values_x[mask].size)
        norm = plt.Normalize(iterations.min(), iterations.max())
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        if idx == last_idx:
            fig.colorbar(sm, cax=cax, label='Iterations')
        else:
            fig.colorbar(sm, cax=cax)
    else:
        axes.scatter(values_x[mask], values_y[mask], alpha=0.5)
    return axes


def get_metrices(path: Path, how='concat'):

    assert how in ['concat', 'per_run'], f'Unknown parameter for \"how\". Has to be one of [concat, per_run] but was {how}'

    path = Path(path)
    metric_files = list(path.rglob('**/all_metric_*.txt'))
    metric_files.sort()

    data = []
    for f in metric_files:
        exp_data = np.loadtxt(f)
        if how == 'concat':
            if len(exp_data.shape) == 1:
                exp_data = np.array([exp_data])
            data.extend(exp_data.tolist())
        elif how == 'per_run':
            data.append(exp_data)
    return data


def plot_pareto_fronts(
        data,
        ax,
        exp_type,
        xl,
        yl
):
    values_x = data[:, 0]
    values_y = data[:, 1]

    # th_list = [0,8]
    maximizator = [-1, -1]
    values = np.asarray([values_x, values_y]).T
    values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
    values = values[values[:, 0].argsort()]
    ax.step(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp_type, where='post')
    ax.set_xlabel(xl, fontweight='bold')
    ax.set_ylabel(yl, fontweight='bold')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.legend()
    return ax


def _get_experiments(path: Path, extension='csv', surrogate=False):
    if surrogate:
        hv_files = path.rglob(f'**/surrogate_hypervolume*.{extension}')
    else:
        hv_files = path.rglob(f'**/hypervolume*.{extension}')
    experiments = [pd.read_csv(f) for f in hv_files]
    return experiments


def plot_aggregated_hypervolume_time_data(data,
                                          axes,
                                          exp_type,
                                          log=False,
                                          resample_hourly=False,
                                          bound_x_lim_left=1,
                                          bound_x_lim_right=24,
                                          plot_legend=True,
                                          color=None,
                                          linestyle='solid'):

    design = {
        'color': color,
        'ls': '-',
        # 'ls': linestyle
    }

    dfs = []
    for i, df in enumerate(data):
        df = df.loc[df.walltime > 0, ['walltime', 'hypervolume']]
        df.loc[:, ['index']] = i

        # df.loc[:, 'walltime'] = pd.to_timedelta(df.loc[:, 'walltime'], unit='sec')
        dfs.append(df)
    dfs = pd.concat(dfs)
    pivot = dfs.pivot(index='walltime', columns='index', values='hypervolume')
    pivot = pivot.ffill(axis=0)
    pivot = pivot[~pivot.isna().any(axis=1)]
    mean = pivot.mean(axis=1).values
    std = pivot.std(axis=1).values

    axes.set_xlim(bound_x_lim_left, bound_x_lim_right)
    if resample_hourly:
        x_label = 'Time in h'
        walltime = pivot.index.values / 3600
        axes.step(walltime, mean,
                  # lw=3.0,
                  label=exp_type, where='post', **design)
    else:
        x_label = 'Time in s'
        walltime = pivot.index.values
        axes.step(walltime, mean,
                  # lw=3.0,
                  label=exp_type, where='post', **design)

    axes.fill_between(walltime, mean + 0.5 * std, mean - 0.5 * std, alpha=0.5, step='post', **design)

    axes.set_xlabel(x_label)
    axes.set_ylabel('Hypervolume')

    if plot_legend:
        axes.legend()

    if log:
        axes.set_xscale('symlog')

    return axes


def plot(result_path: Path,
         output_path: Path,
         is_surrogate: bool,
         x_limit,
         title,
         x_label,
         y_label,
         bench_name,
         benchmark_settings_key: str = None):

    default_benchmark_plotting_settings = {
        'log': False,
    }

    benchmark_settings_key = benchmark_settings_key or result_path.name
    benchmark_settings = load_settings()[benchmark_settings_key]

    benchmark_plotting_settings = default_benchmark_plotting_settings.copy()
    benchmark_plotting_settings.update(benchmark_settings.get('plotting', dict()))

    all_optimizer_settings = load_optimizer_settings()

    result_path = Path(result_path)  # List of available optimizers
    optimizers: List[str] = os.listdir(result_path)
    result_dirs_optimizers = [result_path / opt for opt in optimizers]

    run_id_folder = list(result_path.rglob('**/*.pickle'))
    run_id_folder = [r.parent for r in run_id_folder]

    # check if all metric files exist else compute the missing
    for run_dir in run_id_folder:
        modified_check(Path(run_dir), surrogate=is_surrogate)

    logger.info('Start Aggregated HV Plots')
    fig, axes = plt.subplots(1, figsize=(10, 5))

    # Hypervolume metric over time is saved in file 'hypervolume*.csv' after each experiment
    for result in result_dirs_optimizers:
        optimizer_plotting_settings = all_optimizer_settings[result.name.upper()]['plotting']

        hv_data = _get_experiments(result, extension='csv', surrogate=is_surrogate)
        axes = plot_aggregated_hypervolume_time_data(
            data=hv_data,
            axes=axes,
            exp_type=optimizer_plotting_settings['display_name'],
            # log=benchmark_plotting_settings['log'],
            log=True,
            color=optimizer_plotting_settings['color'],
            linestyle=optimizer_plotting_settings['linestyle'],
            resample_hourly=True,
            bound_x_lim_left=0,
            bound_x_lim_right=x_limit,
            plot_legend=False,
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path / f'hypervolume_curve_{bench_name}_log.pdf', dpi=450)

    export_legend(ax=axes, n_cols=4, filename=output_path / f'hypervolume_curve_{bench_name}_legend.pdf')
    plt.close()

    logger.info('Start Aggregated HV Plots (NOT LOG)')
    fig, axes = plt.subplots(1, figsize=(10, 5))
    # Hypervolume metric over time is saved in file 'hypervolume*.csv' after each experiment
    for result in result_dirs_optimizers:
        optimizer_plotting_settings = all_optimizer_settings[result.name.upper()]['plotting']

        hv_data = _get_experiments(result, extension='csv', surrogate=is_surrogate)
        axes = plot_aggregated_hypervolume_time_data(
            data=hv_data,
            axes=axes,
            exp_type=optimizer_plotting_settings['display_name'],
            log=False,
            color=optimizer_plotting_settings['color'],
            linestyle=optimizer_plotting_settings['linestyle'],
            resample_hourly=True,
            bound_x_lim_left=0,
            bound_x_lim_right=x_limit,
            plot_legend=False,
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path / f'hypervolume_curve_{bench_name}.pdf', dpi=450)

    # export_legend(ax=axes, n_cols=4, filename=output_path / f'hypervolume_curve_{bench_name}_legend.pdf')
    plt.close()

    # Plot attainment surfaces
    n_cols = 3
    n_rows = len(result_dirs_optimizers) // n_cols
    n_rows += 1 if (len(result_dirs_optimizers) % n_cols) != 0 else 0
    f, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 10, n_rows * 8),
                           sharex='all', sharey='all')
    axes = axes.flatten()
    for ax, result_dir_per_optim in zip(axes, result_dirs_optimizers):
        metrics_per_run = get_metrices(result_dir_per_optim, how='per_run')
        k_objectives = metrics_per_run[0].shape[1]
        summary_surfaces = [0, 4, 9]  # the best / median / worst surface.
        # summary_surfaces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # all summary surfaces.

        # get non dominated sets. The used botorch function expects maximization
        non_dominated_sets = []
        scaling = [-1 if target['lower_is_better'] else 1 for target in benchmark_settings['metrics']['target']]

        for run_data in metrics_per_run:
            run_data = scaling * run_data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            run_data = torch.tensor(run_data, device=device).float().contiguous()
            non_dominated = run_data[is_non_dominated(run_data)]
            non_dominated = scaling * non_dominated.cpu().numpy()  # redo the scaling
            non_dominated_sets.append(non_dominated)

        # For each objective: Sort Archives in ascending order according to objective j.
        # Here we assume minimization of all objectives
        scaling = [-1 if not target['lower_is_better'] else 1 for target in benchmark_settings['metrics']['target']]
        non_dominated_sets = [scaling * nds for nds in non_dominated_sets]

        assert k_objectives == 2, 'Currently we only support 2 objectives'

        from MOHPOBenchExperimentUtils.core.attainment_surface import get_attainment_surfaces,\
            linestyle_mapping

        attainment_surfaces = get_attainment_surfaces(non_dominated_sets)

        for i in reversed(summary_surfaces):  # reverse so that the better atts are in the foreground.
            ax.step(attainment_surfaces[i][:, 0], attainment_surfaces[i][:, 1],
                    label=f'S{i + 1}', where='post', ls=linestyle_mapping[i][1])
        ax.set_title(f'{all_optimizer_settings[result_dir_per_optim.name.upper()]["plotting"]["display_name"]}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.savefig(output_path / f'attainment_surfaces_{bench_name}.pdf', dpi=450)
    export_legend(ax=axes[0], n_cols=4, filename=output_path / f'attainment_surfaces_{bench_name}_legend.pdf')
    plt.close()

    logger.info('Start Pareto Front Plots')
    # Final pareto front obtained after each experiment
    fig, axes = plt.subplots(1, figsize=(10, 5))

    for result in result_dirs_optimizers:
        # TODO: DO not append the data.
        #       Either plot attainment surface https://www.cs.bham.ac.uk/~jdk/plot_attainments/
        #       OR look at one pareto front.
        data = get_metrices(result)
        plot_pareto_fronts(np.array(data),
                           axes,
                           exp_type=result.name,
                           xl=x_label,
                           yl=y_label)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'pareto_front_{bench_name}.pdf'), dpi=450)
    plt.close()

    logger.info(f'Start Scatter Plots (1|2)')
    # Scatter plot
    if len(optimizers) > 1:
        fig, axes = plt.subplots(1, len(optimizers), figsize=(5 * len(optimizers), 2*len(optimizers)))
        for idx, (ax, result) in enumerate(zip(axes, result_dirs_optimizers)):
            plot_aggregated_scatter(np.array(get_metrices(result)),
                                    ax, fig,
                                    x_label=x_label,
                                    y_label=y_label,
                                    title=result.name,
                                    idx=idx,
                                    last_idx=len(result_dirs_optimizers) - 1,
                                    iterations_bar=True
                                    )
    elif len(optimizers) == 1:
        fig, axes = plt.subplots(1, figsize=(10, 5))
        assert len(optimizers) == len(result_dirs_optimizers)
        plot_aggregated_scatter(np.array(get_metrices(result_dirs_optimizers[0])),
                                axes, fig,
                                x_label=x_label,
                                y_label=y_label,
                                title=result.name,
                                idx=0,
                                last_idx=len(result_dirs_optimizers) - 1,
                                iterations_bar=True
                                )
    else:
        logger.warning(f'no result exist for bench:{bench_name}')

    plt.tight_layout()
    plt.savefig(output_path / f'scatter_plot_with_iteration_bar_{bench_name}.pdf', dpi=450)
    plt.close()

    logger.info(f'Start Scatter Plots (2|2)')
    # Scatter plot
    if len(optimizers) > 1:
        fig, axes = plt.subplots(1, len(optimizers), figsize=(5 * len(optimizers), 2*len(optimizers)))
        for idx, (ax, result) in enumerate(zip(axes, result_dirs_optimizers)):
            plot_aggregated_scatter(np.array(get_metrices(result)),
                                    ax, fig,
                                    x_label=x_label,
                                    y_label=y_label,
                                    title=result.name,
                                    idx=idx,
                                    last_idx=len(result_dirs_optimizers) - 1,
                                    iterations_bar=False
                                    )
    elif len(optimizers) == 1:
        fig, axes = plt.subplots(1, figsize=(10, 5))
        assert len(optimizers) == len(result_dirs_optimizers)
        plot_aggregated_scatter(np.array(get_metrices(result_dirs_optimizers[0])),
                                axes, fig,
                                x_label=x_label,
                                y_label=y_label,
                                title=result.name,
                                idx=0,
                                last_idx=len(result_dirs_optimizers) - 1,
                                iterations_bar=False
                                )
    else:
        logger.warning(f'no result exist for bench:{bench_name}')
    plt.tight_layout()
    plt.savefig(output_path / f'scatter_plot_{bench_name}.pdf', dpi=450)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path',
                        default='/home/ayushi/PycharmProjects/experiment/MO-HPOBenchExperimentUtils/plot_flower',
                        type=str, nargs='?',
                        help='name of folder where plot files will be dumped')

    #parser.add_argument('--exp_name', default='Random_search', type=str, nargs='?',
    #                    help='name of experiment')
    parser.add_argument('--bench_name', default='Flower', type=str, nargs='?',
                        help='name of experiment')
    parser.add_argument('--title', default='RS_flower_benchmark', type=str, nargs='?',
                        help='title of the plot')
    parser.add_argument('--x_label', default='validation_error', type=str, nargs='?',
                        help='x label')
    parser.add_argument('--y_label', default='log_model_size', type=str, nargs='?',
                        help='y label')
    parser.add_argument('--x_limit', default=12, type=int,
                        help='x_axis limit for hypervolume plot')
    parser.add_argument('--path',
                        default='/home/ayushi/PycharmProjects/experiment/MO-HPOBenchExperimentUtils/mo_cnn_flower-20220602T164559Z-001/mo_cnn_flower',
                        type=str, nargs='?',
                        help='path to result folder')
    parser.add_argument('--is_surrogate',
                        type=str,
                        default='False',
                        help='True if surrogate bench')
    args = parser.parse_args()
    is_surrogate = True if args.is_surrogate == 'True' else False
    logger.debug(f"args is_surrogate:{args.is_surrogate}")
    path = os.path.join(args.output_path)
    os.makedirs(path, exist_ok=True)
    plot(args.path,
         path,
         is_surrogate,
         args.x_limit,
         args.title,
         args.x_label,
         args.y_label,
         args.bench_name)
