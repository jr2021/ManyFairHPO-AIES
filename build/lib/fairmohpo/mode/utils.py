from typing import Any, Dict, List

import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.util as cs_util


def _denormalize_value(hp: CSH.Hyperparameter, normalized_value: float) -> Any:
    is_ordinal = isinstance(hp, CS.OrdinalHyperparameter)
    is_integer = isinstance(hp, CS.UniformIntegerHyperparameter)
    is_categorical = isinstance(hp, CS.CategoricalHyperparameter)
    if is_ordinal or is_categorical:
        choices = hp.sequence if is_ordinal else hp.choices
        ranges = np.arange(start=0, stop=1, step=1 / len(choices))
        param_value = np.array(choices)[normalized_value >= ranges][-1]
    else:
        # rescaling continuous values
        lower = np.log(hp.lower) if hp.log else hp.lower
        upper = np.log(hp.upper) if hp.log else hp.upper
        param_value = lower + (upper - lower) * normalized_value
        param_value = np.exp(param_value) if hp.log else param_value
        param_value = int(np.round(param_value)) if is_integer else float(param_value)

    return param_value


def _denormalize_vector(config_space: CS.ConfigurationSpace, vector: np.ndarray) -> Dict[str, Any]:
    new_config = {
        hp.name: _denormalize_value(hp=hp, normalized_value=vector[i])
        for i, hp in enumerate(config_space.get_hyperparameters())
    }

    # the mapping from unit hypercube to the actual config space may lead to illegal
    # configurations based on conditions defined, which need to be deactivated/removed
    new_config = cs_util.deactivate_inactive_hyperparameters(configuration=new_config, configuration_space=config_space)
    return new_config


def _normalize_value(hp: CSH.Hyperparameter, original_value: float) -> float:
    is_ordinal = isinstance(hp, CS.OrdinalHyperparameter)
    is_categorical = isinstance(hp, CS.CategoricalHyperparameter)
    normalized_value = 0

    if is_ordinal or is_categorical:
        choices = hp.sequence if is_ordinal else hp.choices
        nlevels = len(choices)
        normalized_value = choices.index(original_value) / nlevels
    else:
        lower = np.log(hp.lower) if hp.log else hp.lower
        upper = np.log(hp.upper) if hp.log else hp.upper
        original_value = np.log(original_value) if hp.log else original_value
        normalized_value = (original_value - lower) / (upper - lower)

    return normalized_value


def _normalize_config(config_space: CS.ConfigurationSpace, config: CS.Configuration) -> np.ndarray:
    # the imputation replaces illegal parameter values with their default
    config = cs_util.impute_inactive_values(config)
    vector = np.zeros(len(config_space.get_hyperparameters()))
    for idx, hp in enumerate(config_space.get_hyperparameters()):
        vector[idx] = _normalize_value(hp=hp, original_value=config[hp.name])

    return vector


def _boundary_check(vector: np.ndarray, fix_type: str, rng: np.random.RandomState) -> np.ndarray:
    """
    Checks whether each of the dimensions of the input vector are within [0, 1].
    If not, values of those dimensions are replaced with the type of fix selected.

    if fix_type == 'random', the values are replaced with a random sampling from (0,1)
    if fix_type == 'clip', the values are clipped to the closest limit from {0, 1}

    Parameters
    ----------
    vector : array

    Returns
    -------
    array
    """
    violation_mask = (vector > 1) | (vector < 0)
    if violation_mask.sum == 0:
        return vector
    if fix_type == "random":
        vector[violation_mask] = rng.uniform(low=0.0, high=1.0, size=vector[violation_mask].shape)
    else:
        vector[violation_mask].shape = np.clip(vector[violation_mask].shape, a_min=0, a_max=1)

    return vector


def _initial_sampling(config_space: CS.ConfigurationSpace, pop_size: int) -> np.ndarray:
    # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
    pop_config: List[CS.Configuration]
    if pop_size == 1:
        pop_config = [config_space.sample_configuration(size=pop_size)]
    else:
        pop_config = config_space.sample_configuration(size=pop_size)

    # the population is maintained in a list-of-vector form where each ConfigSpace
    # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
    return np.asarray([_normalize_config(config_space=config_space, config=individual) for individual in pop_config])
