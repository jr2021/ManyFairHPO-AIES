from scripts.visualizations import get_metrices, _hypervolume_evolution_single as hypervolume_eval
from MOHPOBenchExperimentUtils.utils.experiment_utils import save_experiment, save_experiment_df, move_experiment_file
from pathlib import Path
from typing import Union


def save_output(experiment, output_path: Union[str, Path], finished: bool = False, surrogate: bool = False,
                overwrite_experiment: bool = True):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if overwrite_experiment:
        move_experiment_file(output_path / f'{experiment.name}.pickle')
        save_experiment(experiment, output_path / f'{experiment.name}.pickle')

    if finished:
        save_experiment_df(experiment, output_path / f'{experiment.name}_df.csv')

        if surrogate:
            data = hypervolume_eval(experiment, surrogate=True)
            data.to_csv(output_path / f'surrogate_hypervolume_{experiment.name}.csv')
        else:
            data = hypervolume_eval(experiment)
            data.to_csv(output_path / f'hypervolume_{experiment.name}.csv')

        get_metrices(experiment, output_path)
