import glob
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from MOHPOBenchExperimentUtils.utils.experiment_utils import load_experiment
from scripts.log_outputs import save_output


def check_and_compute_missing_files(path, surrogate=False):
    os.chdir(path)
    logger.debug(f'is_surrogate:{surrogate}')
    experiments = glob.glob(f'{path}/*', recursive=True)
    for exp in experiments:
        logger.debug(f"current exp path:{exp}")
        assert glob.glob(f'{exp}/*.pickle'), 'Pickle Not Found'

        # if surrogate is True and surrogate_hypercolume_* doesn't exist,
        # re-calculate hv over time file adding training time
        logger.debug(f"pickle file:{glob.glob(f'{exp}/*.pickle')[0]}")
        logger.debug(f"is surrogate:{surrogate} and file:{glob.glob(f'{exp}/surrogate_hypervolume*.csv')}")
        if (not surrogate and not glob.glob(f'{exp}/hypervolume*.csv')) \
                or not glob.glob(f'{exp}/all_metric*.txt') \
                or (surrogate and not glob.glob(f'{exp}/surrogate_hypervolume*.csv')):
            save_output(experiment=load_experiment(filename=glob.glob(f'{exp}/*.pickle')[0]), output_path=exp,
                        finished=True, surrogate=surrogate)


def modified_check(path: Path, surrogate: bool = False, recompute_all: bool = False):
    logger.debug(f'Call with Path {path} and surrogate: {surrogate}')

    path = path.resolve().absolute()
    # experiment_name = path.parent.parent.name
    experiment_name = 'fairmohpo'
    experiment_pkl = path / f'{experiment_name}.pickle'
    old_experiment_pkl = path / f'OLD-{experiment_name}.pkl'

    hv = path / f'hypervolume_{experiment_name}.csv'
    hv_surrogate = path / f'surrogate_hypervolume_{experiment_name}.csv'
    all_metrics = path / f'all_metric_{experiment_name}.txt'

    assert (experiment_pkl.is_file() and experiment_pkl.exists()) or \
           (old_experiment_pkl.is_file() and old_experiment_pkl.exists()), \
           f'Either {experiment_pkl} or {old_experiment_pkl} has to be present.'

    assert isinstance(surrogate, bool)

    if (recompute_all
            or (not surrogate and not hv.exists())
            or not all_metrics.exists()
            or (surrogate and not hv_surrogate.exists())):

        if recompute_all:
            logger.debug('Going to compute all metrics')

        logger.debug(f'Does all_metric exist: {all_metrics.exists()} - {all_metrics}')
        if not surrogate:
            logger.debug(f'Does HV exist: {hv.exists()} - {hv}')
        else:
            logger.debug(f'Does Surrogate HV exist: {hv_surrogate.exists()} - {hv_surrogate}')

        try:
            experiment = load_experiment(experiment_pkl)
        except Exception as e:
            logger.warning(f'Fallback to old_experiment_pkl: {old_experiment_pkl} due to {e}')
            experiment = load_experiment(old_experiment_pkl)

        save_output(experiment=experiment,
                    overwrite_experiment=False,
                    output_path=path,
                    finished=True,
                    surrogate=surrogate)
    else:
        logger.info('No Missing Files.')


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level='DEBUG')

    parser = ArgumentParser()
    parser.add_argument('--run_directory',
                        default='/work/dlclarge2/muelleph-mo_hpobench/ResultsRQ1/yahpo_surrogate_iaml_iaml_glmnet_1489/random_search/3',
                        type=str,
                        help='directory where a ')

    parser.add_argument('--is_surrogate',
                        action='store_true',
                        default=False,
                        help='True if surrogate bench')

    parser.add_argument('--recompute_all',
                        action='store_true',
                        default=False,
                        help='True if you want to recompute the hv files - just reading in the pickle.')

    args = parser.parse_args()
    run_dir = Path(args.run_directory)
    is_surrogate = args.is_surrogate

    modified_check(path=run_dir, surrogate=is_surrogate, recompute_all=args.recompute_all)
