from typing import List
from typing import Union, Tuple, Dict

import ConfigSpace as CS
import ax
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, IntegerHyperparameter, \
    FloatHyperparameter, Constant
from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace
from loguru import logger


def wrapper_change_hp_in_configspace(
        experiment_name: str,
        old_cs: CS.ConfigurationSpace,
        benchmark_settings: Dict,
        seed: Union[int, None] = None,
) -> CS.ConfigurationSpace:

    cs_modification_settings = benchmark_settings.get('configspace_modification', None)
    if cs_modification_settings is None:
        return old_cs

    # For some experiments, e.g. YAHPO, the settings dict is a nested dict, with the exp name as
    # key. Check if this is the case here and then continue with the remaining part.
    if experiment_name in cs_modification_settings:
        cs_modification_settings = cs_modification_settings[experiment_name]

    hps_to_remove = cs_modification_settings.get('hps_to_remove', None)
    hps_to_replace_with_constant = cs_modification_settings.get('hps_to_replace_with_constant', None)

    new_cs = change_hp_in_configspace(
        old_cs=old_cs,
        hps_to_replace_with_constant=hps_to_replace_with_constant,
        hps_to_remove=hps_to_remove,
        seed=seed
    )
    return new_cs


def change_hp_in_configspace(
        old_cs: CS.ConfigurationSpace,
        hps_to_remove: Union[List[str], None] = None,
        hps_to_replace_with_constant: Union[Dict, None] = None,
        seed: Union[int, None] = None,
) -> CS.ConfigurationSpace:
    """
    Helperfunction: Remove or replace hps in a configuration space.
    In case of replacing, the value from the parameter `hps_to_replace` (Dict) is taken.
    To remove a parameter, specify the parameter name in the list `hps_to_replace`.

    This function updates present conditions in the search space accordingly.

    Parameters
    ----------
    old_cs: CS.ConfigurationSpace
    hps_to_remove: List[str], None
    hps_to_replace_with_constant: Dict[str, Value], None
        You have to provide a Dict that includes for each hp to replace a name (key) and a value
        (value).
    seed: Seed for the ConfigSpace

    Returns
    -------
    CS.ConfigurationSpace
    """

    if hps_to_replace_with_constant is None and hps_to_remove is None:
        return old_cs

    hps_to_remove = hps_to_remove or []
    hps_to_replace_with_constant = hps_to_replace_with_constant or {}

    new_cs = CS.ConfigurationSpace(seed=seed)
    skip_hp_names = hps_to_remove + list(hps_to_replace_with_constant.keys())

    # Add the remaining (untouched) hps.
    non_modified_hps = [hp for hp in old_cs.get_hyperparameters() if hp.name not in skip_hp_names]
    new_cs.add_hyperparameters(non_modified_hps)

    constant_hps = {
        hp_name:
            CS.Constant(
                hp_name,
                hp_value
                    if hp_value != '<DEFAULT>' else
                old_cs.get_hyperparameter(hp_name).default_value
            )
        for hp_name, hp_value in hps_to_replace_with_constant.items()
    }
    new_cs.add_hyperparameters(constant_hps.values())

    # Only add conditions that still have their target/child:
    for condition in old_cs.get_conditions():  # However they also need their parent.
        if condition.child.name in new_cs.get_hyperparameter_names():
            if condition.parent.name in skip_hp_names:
                condition.parent = constant_hps[condition.parent.name]
            if condition.child.name in skip_hp_names:
                condition.child = constant_hps[condition.child.name]
            try:
                new_cs.add_condition(condition)
            except ValueError as e:
                logger.warning(
                    f'We were not able to add the following condition: {condition} due to Exception: {e}'
                )
    return new_cs


def get_parameter_type_for_categorical_choices(values: Union[Tuple, List]) -> Union[ParameterType, int]:
    '''

    This Method determines type for CS.CategoricalHyperparameter by checking the values in choices
    AX space use ChoiceParameter for Categorical type and parameter type is the type of values in choices
    :param values: values in Categorical type
    :return: Parameter Type

    '''
    if all(type(item) is bool for item in values):
        return ParameterType.BOOL
    elif all(type(item) is int for item in values):
        return ParameterType.INT
    elif all(type(item) is float for item in values):
        return ParameterType.FLOAT
    elif all(type(item) is str for item in values):
        return ParameterType.STRING
    else:
        item = values[0]
        raise ValueError(f'Unknown type: {type(item)}')


def get_parameter_type_constant(item: Union[int, str, float]) -> Union[ParameterType, int]:
    '''
    This Method determines type for CS.CategoricalHyperparameter by checking the type of value
    AX space use ChoiceParameter for Categorical type and parameter type is the type of values in choices
    :param values: values in Categorical type
    :return: Parameter Type
    '''
    if type(item) is bool:
        return ParameterType.BOOL
    elif type(item) is int:
        return ParameterType.INT
    elif type(item) is float:
        return ParameterType.FLOAT
    elif type(item) is str:
        return ParameterType.STRING
    else:
        raise ValueError(f'Unknown type: {type(item)}')


def adapt_axspace_to_config(config: Dict, config_space_keys):
    '''

    HPOBench takes CS.Configuration or Dict type for Configuration
    AX space is Dict type with a few additional parameters such as fidelity and id
    This function removes AX specific parameters
    Args:
        config: Dict which represents AX search space

    Returns:
        Dict: After removing AX specific keys
    '''
    cs_config_keys = [key for key in config.keys() if key not in config_space_keys]
    for item in cs_config_keys:
        config.pop(item)
    return config


def adapt_configspace_configuration_to_ax_space(params, search_space, keys_to_ignore=[]):
    '''
    AX(0.1.18) doesn't support conditional parameters
    This is a hecky way that adapts the sampled configuartion and add the missing keys with default values:
    This is done just before the evaluation -- doesn't change the optimizer
    Call to objective function in HPOBench handles it by giving a warning for inactive parameters and deactivating them
    '''
    config_hps = search_space.config_space.get_hyperparameters_dict()
    for key, value in search_space.ax_space.parameters.items():
        if key in keys_to_ignore:
            continue
        if key not in params:
            params[key] = config_hps[key].default_value
            logger.warning(f"adding missing {key} parameters with default values:{config_hps[key].default_value}")
        # added for categorical as a safeguard
        # if value.parameter_type is ParameterType.BOOL and type(params[key]) is not bool:
        #     params[key] = bool(params[key])
        #     logger.warning(f"converting boolean parameter {key} to boolean type")
    return params


def convert_config_space_to_ax_space(config: CS.ConfigurationSpace, max_budget=25, fidelities: List[Dict] = None) \
        -> ax.SearchSpace:
    """
    Returns: AX search space
    """
    hyperparameters = config.get_hyperparameters_dict()
    ax_space_dict = {}

    for key, value in hyperparameters.items():
        if isinstance(value, CategoricalHyperparameter):
            values = value.choices
            if not isinstance(values, List):
                values = list(values)
            if len(values) == 1:
                parameter_type = get_parameter_type_constant(values[0])
                ax_space_dict[key] = FixedParameter(name=value.name, parameter_type=parameter_type, value=values[0])
            else:
                parameter_type = get_parameter_type_for_categorical_choices(values)
                ax_space_dict[key] = ChoiceParameter(name=value.name,
                                                     values=values,
                                                     parameter_type=parameter_type)

        elif isinstance(value, OrdinalHyperparameter):
            values = value.sequence
            if not isinstance(values, List):
                values = list(values)
            parameter_type = get_parameter_type_for_categorical_choices(values)
            ax_space_dict[key] = ChoiceParameter(name=value.name,
                                                 values=values,
                                                 parameter_type=parameter_type)

        elif isinstance(value, IntegerHyperparameter):
            ax_space_dict[key] = RangeParameter(name=value.name,
                                                parameter_type=ParameterType.INT,
                                                lower=value.lower,
                                                upper=value.upper,
                                                log_scale=value.log)

        elif isinstance(value, FloatHyperparameter):
            ax_space_dict[key] = RangeParameter(name=value.name,
                                                parameter_type=ParameterType.FLOAT,
                                                lower=value.lower,
                                                upper=value.upper,
                                                log_scale=value.log)

        elif type(value) is Constant:
            parameter_type = get_parameter_type_constant(value.value)
            ax_space_dict[key] = FixedParameter(name=value.name,
                                                parameter_type=parameter_type,
                                                value=value.value)

    # TODO: Why do we need that? Only Single Fidelity?
    if fidelities is not None:
        for fidelity in fidelities:
            parameter_type = get_parameter_type_constant(fidelity['limits'][1])
            ax_space_dict[fidelity['name']] = FixedParameter(fidelity['name'], parameter_type, fidelity['limits'][1])
    else:
        # TODO; Why do we need this one?
        ax_space_dict['budget'] = FixedParameter('budget', ParameterType.INT, max_budget)

    # Need this one for Ax: It has problems to schedule the same configuration multiple times
    ax_space_dict['id'] = FixedParameter('id', ParameterType.STRING, 'dummy')

    ax_space = SearchSpace(parameters=list(ax_space_dict.values()))
    return ax_space


def convert_config_space_to_numeric_space(config: CS.ConfigurationSpace):
    """
    Returns: AX search space
    """
    from typing import List
    cs = CS.ConfigurationSpace()
    hyperparameters = config.get_hyperparameters_dict()
    for key, value in hyperparameters.items():
        if type(value) is CS.hyperparameters.CategoricalHyperparameter:
            if not isinstance(value.choices, List):
                values = list(value.choices)
            cs.add_hyperparameter(CS.UniformIntegerHyperparameter(name=value.name,
                                                                           lower=0,
                                                                           upper=len(values) - 1,
                                                                           default_value=values.index(
                                                                               value.default_value)))
        elif type(value) is CS.hyperparameters.UniformIntegerHyperparameter or \
                type(value) is CS.hyperparameters.UniformFloatHyperparameter:
            cs.add_hyperparameter(config.get_hyperparameter(key))
    return cs


def adapt_numeric_configuration_to_ax_space(params, search_space, keys_to_ignore=[]):
    '''

    AX(0.1.18) doesn't support conditional parameters
    This is a hecky way that adapts the sampled configuartion and add the missing keys with default values:
    This is done just before the evaluation -- doesn't change the optimizer
    Call to objective function in HPOBench handles it by giving a warning for inactive parameters and deactivating them

    '''

    from ax import ParameterType, ChoiceParameter
    ax_space = search_space.ax_space
    logger.debug(f'ax space:{ax_space}')

    for key, value in ax_space.parameters.items():

        print(type(value))
        if key in keys_to_ignore:
            continue
        if type(value) is ChoiceParameter:
            if value.parameter_type is ParameterType.BOOL:
                assert params[key] in (0, 1)
                params[key] = bool(params[key])
            else:
                assert params[key] in range(len(value.values))
                params[key] = value.values[params[key]]

    logger.debug(f"param after adapting:{params}")
    return params
