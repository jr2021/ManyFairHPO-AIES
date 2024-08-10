import pickle
import shutil
from pathlib import Path
from typing import Any, Union

from ax import Experiment
from ax.core.base_trial import TrialStatus

from loguru import logger
from importlib import import_module


class OutOfTimeException(Exception):
    # Custom exception for easy handling of timeout
    pass


def load_object(import_name: str, import_from: str) -> Any:
    """Helperfunction:
    Dynamically load an python object.

    Corresponds to the line of code:
    from `import_from` import `import_name`
    """

    logger.debug(f'Try to execute command: from {import_from} import {import_name}')
    module = import_module(import_from)
    obj = getattr(module, import_name)
    return obj


def load_optimizer(optimizer_name: str, import_from: str) -> Any:
    """
    Load the optimizer object.

    Parameters
    ----------
    benchmark_name : str
    import_from : str

    Returns
    -------
    Optimizer
    """
    import_str = 'MOHPOBenchExperimentUtils.methods.' + import_from
    optimizer_obj = load_object(optimizer_name, import_str)
    logger.debug(f'Optimizer {optimizer_name} successfully loaded')

    return optimizer_obj


def move_experiment_file(filename: Union[str, Path]):
    """
    We have encountered errors on the cluster due to memory limits. This is an additional safety function.
    If an experiment pickle file already exists,
    rename it to OLD-<filename>.pkl. If an file OLD-<filename> exists delete that.

    Parameters
    ----------
    filename: Path
        path to the experiment pickle.
    """

    if not filename.exists():
        return

    name = 'OLD-' + filename.name.replace('.pickle', '.pkl')
    backup_file = filename.parent / name

    if backup_file.exists():
        backup_file.unlink()

    shutil.move(filename, backup_file)


def save_experiment(experiment: Experiment, filename: Union[str, Path]):
    with open(filename, 'wb') as file:
        pickle.dump(experiment, file, protocol=-1)


def save_experiment_df(experiment: Experiment, filename: Union[str, Path]):
    # There is a bug that if a run was scheduled but not completed, fetch df tries to reschedule them.
    # Therefore, we solve this by only fetching the data of completed runs.
    ind_completed_runs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
    df = experiment.fetch_trials_data(trial_indices=ind_completed_runs).df
    df.to_csv(filename)


def load_experiment(filename: Union[str, Path]):
    with open(filename, 'rb') as file:
        experiment = pickle.load(file)
    return experiment
