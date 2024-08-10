import yaml
from pathlib import Path
from typing import Dict, List, Union
from copy import deepcopy


def __load_yaml_file(yaml_file: Union[str, Path]) -> Dict:
    yaml_file = Path(yaml_file)
    with yaml_file.open('r') as fh:
        payload = yaml.load(fh, yaml.FullLoader)
    return payload


def _load_yahpo_settings(yahpo_surrogates_file: Path) -> Dict:
    """
    Because yahpo has a lot of runs that are tested on different task ids and we are too lazy to write everything down
    we define run settings for classes of yahpo experiments.
    We add in the header of an experiment, the name of the scenarions (learner) and instances (task_ids) that are
    available for this setting.
    """
    yahpo_settings = __load_yaml_file(yahpo_surrogates_file)

    # Create the yahpo settings on the fly
    all_yahpo_settings = {}
    for name, setting in yahpo_settings.items():
        # Occurs for the IAML data sets
        if name == 'yahpo_surrogate_iaml':
            # This dict maps optimizer to instance. We use these information to set the correct
            # benchmark parameters.
            available_optimizers_datasets = setting['available_optimizers_datasets']

            for scenario in available_optimizers_datasets.keys():
                for instance in available_optimizers_datasets[scenario]:
                    # The correct name for this setting.
                    run_name = f'{name}_{scenario}_{instance}'

                    # The rest of dictionary corresponds to the default settings for this kind of run
                    # Basically, we only have to adapt the benchmark settings
                    run_setting = deepcopy(setting)
                    run_setting['benchmark_parameters'].update({'scenario': scenario, 'instance': instance})
                    del run_setting['available_optimizers_datasets']
                    all_yahpo_settings[run_name] = run_setting

        elif 'yahpo_surrogate_fair' in name:
            # This dict maps optimizer to instance and targets. We use that information to set the correct
            # benchmark parameters.
            mapping_info = setting['fair_mapping_opt_taskid_target']
            for scenario, instance, targets in mapping_info:
                run_name = f'{name}_{scenario}_{instance}'

                # The rest of dictionary corresponds to the default settings for this kind of run
                # We have to adapt the benchmark settings similar to the step above
                run_setting = deepcopy(setting)
                run_setting['benchmark_parameters'].update({'scenario': scenario, 'instance': instance})

                # However, we have this time also different metrics per scenario-instance pair.
                metrics = run_setting['metrics']
                metrics['target'][1]['name'] = targets[1]
                metrics['additional'] = [entry for entry in metrics['additional'] if entry['name'] != targets[1]]
                run_setting['metrics'] = metrics

                del run_setting['fair_mapping_opt_taskid_target']
                all_yahpo_settings[run_name] = run_setting

        # Of course it can happen that at some point someone writes a normal configuration into the EXP_YAHPO_SURROGATES
        # file. Therefore, i add this else clause :D
        else:
            # It is a normal configuration that can be used directly.
            all_yahpo_settings[name] = setting
    return all_yahpo_settings


def load_settings() -> Dict:
    """ Load the settings from the settings yaml. """
    # --------------------------- LOAD THE NORMAL SETTING FILE ---------------------------------------------------------
    settings_file = Path(__file__).absolute().parent.parent / 'settings' / 'EXP_SETTINGS.yaml'
    settings = __load_yaml_file(settings_file)

    # --------------------------- LOAD THE ML TABULAR SETTING FILE -----------------------------------------------------
    ml_settings_file = Path(__file__).absolute().parent.parent / 'settings' / 'EXP_ML_TABULAR.yaml'
    ml_settings = __load_yaml_file(ml_settings_file)

    # --------------------------- LOAD THE YAHPO SETTING FILE ----------------------------------------------------------
    yahpo_surrogates_file = Path(__file__).absolute().parent.parent / 'settings' / 'EXP_YAHPO_SURROGATES.yaml'
    yahpo_settings = _load_yahpo_settings(yahpo_surrogates_file)

    return {**settings, **ml_settings, **yahpo_settings}


def load_optimizer_settings() -> Dict:
    """ Load the optimizer settings from yaml. """
    opt_settings_file = Path(__file__).absolute().parent.parent / 'settings' / 'OPTIMIZER_SETTINGS.yaml'
    opt_settings = __load_yaml_file(opt_settings_file)
    return opt_settings


def get_experiment_names() -> List[str]:
    """ Return the list of available experiments. """
    settings = load_settings()
    experiment_names = list(settings.keys())
    return experiment_names
