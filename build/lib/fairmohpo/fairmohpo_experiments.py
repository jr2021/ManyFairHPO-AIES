import argparse
import signal
import time
import os

from pathlib import Path
from pprint import pformat

import ax
from hpobench.config import config_file
from loguru import logger

from MOHPOBenchExperimentUtils import save_experiment
from MOHPOBenchExperimentUtils.methods.pymoo_.pymoo_base import HPOBenchProblem
from MOHPOBenchExperimentUtils.problems.hpobench_benchmarks import get_experiment
from MOHPOBenchExperimentUtils.utils.experiment_utils import OutOfTimeException, load_optimizer
from MOHPOBenchExperimentUtils.utils.hpobench_utils import load_benchmark
from MOHPOBenchExperimentUtils.utils.settings_utils import load_settings, load_optimizer_settings

from scripts.log_outputs import save_output
from fairmohpo_utils import (
    get_objective_names,
    get_fairness_objective_settings,
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='fairmohpo_rf')
    parser.add_argument('--optimizer_name', type=str, default='NSGA_II_DEFAULT')
    parser.add_argument('--objective_name', type=str, default='f1')
    parser.add_argument('--model_name', type=str, default='rf')
    parser.add_argument('--dataset_name', type=str, default='compas')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_path', type=str, nargs='?', default='logs',
                        help='name of folder where files will be dumped')
    args = parser.parse_args()

    # -------------------- CREATE THE OUTPUT DIRECTORY -----------------------------------------------------------------
    output_path = Path(args.output_path) / args.experiment_name / args.optimizer_name / args.model_name / args.dataset_name / str(args.seed)
    if output_path.exists() and output_path.is_dir() and any(os.scandir(output_path)):
        print('Output path: ', output_path)
        raise ValueError('The Directory already exists and is not empty.')
    Path(output_path).mkdir(exist_ok=True, parents=True)

    # -------------------- PREPARE SETTINGS -------------------------------------------------------------------------------
    all_settings = load_settings()

    settings = all_settings[args.experiment_name]
    
    settings['objective_name'] = args.objective_name
    settings['objectives'] = get_objective_names(args.objective_name)
    settings['model_name'] = args.model_name
    settings['dataset_name'] = args.dataset_name
    settings['seed'] = args.seed

    for objective in settings['objectives'][1:]:
        settings['metrics']['target'].append(get_fairness_objective_settings(objective))

    for metric in settings['metrics']['target']:
        settings['metrics']['additional'].remove(
            {'name': metric['name'], 'lower_is_better': metric['lower_is_better']}
        )
    num_objectives = len(settings['metrics']['target'])

    all_optimizer_settings = load_optimizer_settings()
    optimizer_settings = all_optimizer_settings[args.optimizer_name]
    logger.info(
        f'Settings: \n'
        f'{pformat(settings)} \n\n'
        f'Optimizer Settings: \n'
        f'{pformat(optimizer_settings)}'
    )

    tae_limit = settings['optimization'].get('tae_limit', None)
    wallclock_limit_in_s = settings['optimization'].get('wallclock_limit_in_s', 4 * 24 * 3600)
    estimated_cost_limit_in_s = settings['optimization'].get('estimated_cost_limit_in_s', 4 * 24 * 3600)
    is_surrogate = settings['optimization'].get('is_surrogate', False)

    def signal_handler(sig, frame):
        save_experiment(experiment, f'{experiment.name}.pickle')
        logger.info('Job is being cancelled')
        raise OutOfTimeException

    signal.signal(signal.SIGALRM, signal_handler)  # register the handler
    signal.alarm(wallclock_limit_in_s - (10 * 60))

    # -------------------- Initialize Benchmark + Experiment -----------------------------------------------------------  
    benchmark_object = load_benchmark(**settings['import'])
    benchmark = benchmark_object(
        container_source=config_file.container_source,
        rng=args.seed,
        **settings.get('benchmark_parameters', {}),
    )
    benchmark.init(settings, seed=args.seed)

    experiment = get_experiment(
        name=args.experiment_name,
        settings=settings,
        rng=args.seed,
        socket_id=benchmark.socket_id,
        output_path=output_path,
        tae_limit=tae_limit,
        wallclock_limit_in_s=wallclock_limit_in_s,
        estimated_cost_limit_in_s=estimated_cost_limit_in_s,
        is_surrogate=is_surrogate,
    )

    # -------------------- INIT Optimizer-------------------------------------------------------------------------------
    optimizer = Pymoo_NSGA2(
        optimizer_settings=optimizer_settings, benchmark_settings=settings,
        configspace=experiment.cs_search_space, wallclock_limit_in_s=wallclock_limit_in_s,
    )
    optimizer.init()

    # -------------------- DEFINE PYMOO PROBLEM ------------------------------------------------------------------------
    _, lower_limits, upper_limits, constant_values = optimizer.get_pymoo_cs()

    problem = HPOBenchProblem(
        experiment=experiment, lower_limits=lower_limits, upper_limits=upper_limits,
        num_hp=len(lower_limits), fidelity_settings=settings['fidelity'],
        constant_values=constant_values, num_objectives=num_objectives, num_constraints=0
    )

    # -------------------- RUN OPTIMIZER -------------------------------------------------------------------------------
    optimizer.setup(problem=problem, seed=args.seed)
    experiment.start_timer()
    saved = False

    logger.info('Optimizer running')

    # try:
    optimizer.run()

    # except (OutOfTimeException, TimeoutError) as err:
    #     save_output(experiment, output_path, finished=True)
    #     saved = True
    #     logger.info("Timeout Reached: {}", err)

    # except (Exception, KeyboardInterrupt, BaseException) as err:
    #     save_output(experiment, output_path, finished=False)
    #     saved = True
    #     logger.info("catching error and checkpointing the result: {}", err)

    # if not saved:
    #     save_output(experiment, output_path, finished=True)

    # try:
    #     benchmark.__del__()
    # except:
    #     pass

    # logger.info(f'Done.')
