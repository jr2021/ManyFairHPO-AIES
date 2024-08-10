import pickle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from MOHPOBenchExperimentUtils.utils.experiment_utils import load_experiment
from ax.core.base_trial import TrialStatus
from MOHPOBenchExperimentUtils.utils.hpobench_utils import HPOBenchMetrics
from dask.distributed import Client, progress
from MOHPOBenchExperimentUtils import logger


def read_csv_file(iteration_number: int, pickle_file: Path, n_files: int, old_data: pd.DataFrame):
    from MOHPOBenchExperimentUtils import logger as logger_2

    pickle_file = pickle_file.resolve().absolute()
    run_id = str(pickle_file.parent.name)
    optimizer = pickle_file.parent.parent.name
    benchmark = pickle_file.parent.parent.parent.name

    name = 'OLD-' + pickle_file.name.replace('.pickle', '.pkl')
    backup_file = pickle_file.parent / name

    query = f'benchmark == \'{benchmark}\' & optimizer == \'{optimizer}\' & run_id == \'{run_id}\''
    if old_data is not None and len(df.query(query)) != 0:
        logger_2.info(f'[{iteration_number + 1:06}|{n_files:06}] {benchmark} {optimizer} {run_id}: '
                      f' SKIP (already collected)')
        return None

    try:
        experiment = load_experiment(pickle_file)
    except (EOFError, pickle.UnpicklingError) as err:
        logger.info(f'Error during loading experiment file {err}.')
        if backup_file.exists():
            logger.info(f'Try to load backup file {backup_file}')
            experiment = load_experiment(backup_file)  # hopefully this should not crash.
        else:
            logger.info('No Backup File: Return 0.')
            writer.writerow([benchmark, optimizer, run_id, 0, 0, 0, str(pickle_file)])
            return [benchmark, optimizer, run_id, 0, 0, 0, str(pickle_file)]

    total_trials = len(experiment.trials)

    # not using time_completed for last run because it doesn't exist for some runs
    # maybe in case of run crash
    total_wallclocktime = (experiment.trials[total_trials - 1].time_created - experiment.trials[
        0].time_created).total_seconds()

    ind_completed_runs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
    run_df = experiment.fetch_trials_data(trial_indices=ind_completed_runs).df
    cost_df = run_df.loc[run_df.loc[:, 'metric_name'] == HPOBenchMetrics.COST.value]
    returned_costs = cost_df.loc[:, 'mean'].sum()

    logger_2.info(f'[{iteration_number + 1:06}|{n_files:06}] {benchmark} {optimizer} {run_id}: '
                  f'#Trials: {total_trials:10}'
                  f' - Runtime: {total_wallclocktime:10.1f}'
                  f' - Costs: {returned_costs:10.1f}')

    try:
        del experiment
        del logger_2
    except Exception as e:
        print(f'Exception: {e}')
        pass

    return [benchmark, optimizer, run_id, total_trials, total_wallclocktime, returned_costs, str(pickle_file)]


def get_run_stats(result_dir: Path, output_file: Path, bench=None, old_data=None, num_cpus=1):

    pickle_files = list(result_dir.rglob('*.pickle'))
    logger.info(f'Found a total of {len(pickle_files)}')
    if bench is not None:
        pickle_files = [p for p in pickle_files if bench in str(p)]
        logger.info(f'Selected {len(pickle_files)} of them.')

    pickle_files.sort()
    import multiprocessing

    with multiprocessing.Pool(processes=num_cpus) as pool:
        params = [(i, pickle_file, len(pickle_files), old_data) for i, pickle_file in enumerate(pickle_files)]
        results = pool.starmap(read_csv_file, params)

    results = [r for r in results if r is not None]

    df = pd.DataFrame(results, columns=['benchmark', 'optimizer', 'run_id', 'n_trials', 'time', 'cost', 'path'])
    df = pd.concat([df, old_data])
    df = df.sort_values(by=['benchmark', 'optimizer', 'run_id'])

    df.to_csv(output_file)
    logger.info(f'Saved Results to {output_file}')


if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--run_directory',
                        default='/work/dlclarge2/muelleph-mo_hpobench/ResultsRQ1',
                        type=str,
                        help='directory where all runs are stored')
    parser.add_argument('--output_directory',
                        default='/work/dlclarge2/muelleph-mo_hpobench/run_stats',
                        type=str,
                        help='output path')
    parser.add_argument('--bench_name',
                        default=None,
                        type=str,
                        help='if stats run needs to be evaluated for a single bench')
    parser.add_argument('--num_cpus',
                        default=1,
                        type=int,
                        help='Number of CPUS available. Defaults to 1.')

    args = parser.parse_args()
    run_dir = Path(args.run_directory)
    output_dir = Path(args.output_directory)
    output_dir = output_dir.resolve().absolute()
    output_dir.mkdir(exist_ok=True)

    bench = args.bench_name

    if bench is not None:
        output_file = output_dir / f'stats_{bench}.csv'
    else:
        output_file = output_dir / f'stats.csv'

    df = None
    header = ['benchmark', 'optimizer', 'run_id', 'n_trials', 'time', 'cost', 'path']
    new_file = not output_file.exists()

    if not new_file:
        logger.info('Found an existing csv file. Read that in. ')
        df = pd.read_csv(output_file, names=header, skiprows=1)

    get_run_stats(result_dir=run_dir, output_file=output_file, bench=bench, old_data=df, num_cpus=args.num_cpus)

    logger.info(f'Finished writing to {output_file}')
