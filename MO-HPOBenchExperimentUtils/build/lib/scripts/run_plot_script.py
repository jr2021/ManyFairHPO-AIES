from scripts.plot_mo_metric import plot
import argparse
import os
from MOHPOBenchExperimentUtils import logger
from MOHPOBenchExperimentUtils.utils.settings_utils import load_settings, load_optimizer_settings
from pathlib import Path


def plot_all_exp(args):

    all_settings = load_settings()

    for key, benchmark_settings in all_settings.items():
        if args.use_only_benchmark != 'all' and args.use_only_benchmark != key:
            continue
        logger.info(f'Start to Plot Experiment: {key}')

        is_surrogate = benchmark_settings['optimization']['is_surrogate']

        if not is_surrogate:
            x_limit = benchmark_settings['optimization']['wallclock_limit_in_s'] / 3600
        else:
            x_limit = benchmark_settings['optimization']['estimated_cost_limit_in_s'] / 3600

        x_label = benchmark_settings['metrics']['target'][0]['name']
        y_label = benchmark_settings['metrics']['target'][1]['name']

        output_dir = os.path.join(args.output_path, key)
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f'bench:{key}, is_surrogate:{is_surrogate},y_label:{y_label},x label:{x_label},x_limit:{x_limit}')

        plot(
            result_path=Path(args.result_dir) / key,
            output_path=Path(output_dir),
            is_surrogate=is_surrogate,
            x_limit=x_limit,
            title=key,
            x_label=x_label,
            y_label=y_label,
            bench_name=key,
            benchmark_settings_key=key,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path',
                        default='/home/ayushi/PycharmProjects/experiment/MO-HPOBenchExperimentUtils/plots/',
                        type=str, nargs='?',
                        help='name of folder where plot files will be dumped')
    parser.add_argument('--result_dir', default='/home/ayushi/PycharmProjects/experiment/MO-HPOBenchExperimentUtils/mo_cnn_flower-20220602T164559Z-001/', type=str, nargs='?',
                        help='name of experiment')
    parser.add_argument('--use_only_benchmark', type=str, default='all')
    args = parser.parse_args()
    plot_all_exp(args)
