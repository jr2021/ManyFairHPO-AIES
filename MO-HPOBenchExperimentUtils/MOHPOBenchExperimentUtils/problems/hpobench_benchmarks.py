from typing import Union, Dict

import ConfigSpace as CS
import ax
import numpy as np
from hpobench.config import config_file
from hpobench.container.client_abstract_benchmark import AbstractMOBenchmarkClient, AbstractBenchmarkClient
from loguru import logger
from time import time
from typing import List
from pathlib import Path

from MOHPOBenchExperimentUtils.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
from MOHPOBenchExperimentUtils.core.target_normalization import TargetScaler, get_scaler
from MOHPOBenchExperimentUtils.utils.ax_utils import get_ax_metrics_from_metric_dict
from MOHPOBenchExperimentUtils.utils.hpobench_utils import load_benchmark, HPOBenchMetrics
from MOHPOBenchExperimentUtils.utils.search_space_utils import convert_config_space_to_ax_space, \
    wrapper_change_hp_in_configspace


class WrappedBenchmark:
    def __init__(self, settings: Dict, rng: int, socket_id: str = None):

        self.benchmark: Union[AbstractMOBenchmarkClient, AbstractBenchmarkClient, None] = None
        self.init_benchmark(settings, rng, socket_id)

        # If socket id was None, then initializing the benchmark creates a new socket id.
        self.socket_id = self.benchmark.socket_id
        self.settings = settings
        self.rng = rng
        self.start_time = time()

        # In some cases, it is interesting to scale the output of the benchmarks to a certain range.
        # E.g. the targets of a benchmarks are between [0, 100] and we want to scale it with a
        # MinMax scaler to [0,1], because the HPO algorithms are reliant on a certain scale.
        default_scaler = {'algorithm': 'NoOPScaler'}
        self.objective_scaler: Dict[str, TargetScaler] = {
            target['name']: get_scaler(**target.get('normalize', default_scaler))
            for target in self.settings['metrics']['target']
        }

    def init_benchmark(self, settings: Dict, rng: int, socket_id: str = None) -> None:
        benchmark_object = load_benchmark(**settings['import'])
        self.benchmark: Union[AbstractMOBenchmarkClient, AbstractBenchmarkClient] = benchmark_object(
            container_source=config_file.container_source,
            rng=rng,
            **settings.get('benchmark_parameters', {}),
            socket_id=socket_id,
        )

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchmark.get_configuration_space(seed=seed)

    def eval_function(self, config, fidelity=None) -> Dict:
        start_time_config = time() - self.start_time

        # Extract fidelities from config. If it is not present, replace it with the default max value.
        fidelities = {}
        for f in self.settings['fidelity']:
            fidelity_name, max_budget = f['name'], f['limits'][1]
            fidelities[fidelity_name] = config.get(fidelity_name, max_budget)
            if fidelity_name in config:
                del config[fidelity_name]

        if fidelity is not None:
            fidelities.update(fidelity)

        # Remove the ax id from the configuration -
        # We need the ID field in the configuration for MOEA/D. There, it is possible that the same
        # configuration is evaluated multiple times. That leads to an error in ax.
        if 'id' in config:
            config.pop('id')

        # Cast the ax configuration to a cs.configuration
        # TODO: It might be necessary to cast the ax config space to cs config space.
        if not isinstance(config, (dict, Dict, CS.Configuration)):
            logger.warning(f'Config Type: {type(config)}')

        # Connect to the benchmark and query the objective function
        benchmark_object = load_benchmark(**self.settings['import'])
        benchmark: Union[AbstractMOBenchmarkClient, AbstractBenchmarkClient] = benchmark_object(
            container_source=config_file.container_source,
            rng=self.rng,
            socket_id=self.socket_id,
            **self.settings.get('benchmark_parameters', {})
        )
        benchmark.init(self.settings, seed=self.rng)

        logger.debug(f'Query config: {config} and fidelity: {fidelities}')
        result_dict = benchmark.objective_function(
            configuration=config,
            fidelity=fidelities,
            rng=self.rng
        )

        # Apply target scaling if available
        scaled_func_values = {
            obj: scaler.transform(result_dict['function_value'][obj])
            for obj, scaler in self.objective_scaler.items()
        }
        result_dict['function_value'] = scaled_func_values

        def flatten_dict(hierarchical_dict) -> Dict:
            """Helper function to flatten a dict.
            Example: 
                >>> input = {'x': 12, 'y': {'a': 1, 'b': 2, 'c': {'g': 'test'}}}
                >>> output = flatten_dict(input)
                >>> print(output) 
                {'x': 12, 'a': 1, 'b': 2, 'g': 'test'}
                
            Parameters
            ----------
                hierarchical_dict: Dict

            Returns
            -------
                Dict
            """
            flat_dict = {}
            for key, value in hierarchical_dict.items():
                if isinstance(value, Dict):
                    sub_dict = flatten_dict(value)
                    flat_dict.update(sub_dict)
                elif isinstance(value, (List, np.ndarray)):
                    # f"Ignoring {key} to prevent later issues with ax"
                    pass
                else:
                    flat_dict[key] = value
            return flat_dict

        # HPOBench returns a nested dictionary, however, ax expects a flat dict.
        # (We currently have only dictionaries in the hpobench of a maximal depth of 2)
        flat_dict = flatten_dict(result_dict['info'])
        flat_dict.update(fidelities)
        flat_dict.update({
            HPOBenchMetrics.COST.value: result_dict['cost'],
            HPOBenchMetrics.WALLCLOCK_CONFIG_START.value: start_time_config,
            HPOBenchMetrics.WALLCLOCK_CONFIG_END.value: time() - self.start_time
        })
        flat_dict.update(result_dict['function_value'])

        # Remove string values from the dictionary. Ax can't handle that, unfortunately.
        return_contains_str = any(isinstance(v, str) for v in flat_dict.values())
        if return_contains_str:
            for k, v in flat_dict.items():
                if isinstance(v, str):
                    logger.warning(
                        f'Ax can\'t handle str return types. Remove key {k} and value {v} from the result dict.'
                    )
            flat_dict = {k: v for k, v in flat_dict.items() if not isinstance(v, str)}

        # Ax expects the values of each key to be a tuple of size two with values (mean, sem).
        result_dict_ax = {key: (value, 0) for key, value in flat_dict.items()}
        logger.debug(f'Register Result: {result_dict_ax}')
        return result_dict_ax

    def __del__(self):
        logger.debug('Wrapped benchmark del called')

        try:
            self.benchmark.__del__()
        except Exception:
            logger.debug('Wrapped benchmark del called and crashed')

    def __getstate__(self):
        logger.debug('Call the \'__getstate__()\' function.')

        # We delete the benchmark object before pickle the object
        self.benchmark = None
        return self.__dict__

    def __setstate__(self, state):
        # But we don't initialize the benchmark object!
        logger.debug('Call the \'__setstate__()\' function.')

        if 'benchmark' in state:
            state['benchmark'] = None

        self.__dict__ = state


def get_experiment(name: str, settings: Dict, rng: int,
                   output_path: Union[str, Path],
                   tae_limit: Union[int, None],
                   wallclock_limit_in_s: Union[int, float],
                   estimated_cost_limit_in_s: Union[int, float],
                   is_surrogate: bool,
                   socket_id: Union[str, None] = None) -> MultiObjectiveSimpleExperiment:

    # Initialize the benchmark
    wrapped_benchmark = WrappedBenchmark(settings=settings, rng=rng, socket_id=socket_id)

    # Initialize the experiment
    cs = wrapped_benchmark.get_configuration_space(seed=rng)
    cs = wrapper_change_hp_in_configspace(
        experiment_name=name, old_cs=cs, benchmark_settings=settings, seed=rng,
    )

    ax_cs = convert_config_space_to_ax_space(cs, fidelities=settings['fidelity'])

    multi_objective, objective_thresholds, extra_metrics = get_ax_metrics_from_metric_dict(
        settings['metrics']
    )

    optimization_config = ax.MultiObjectiveOptimizationConfig(
        objective=multi_objective,
        objective_thresholds=objective_thresholds,
    )

    multi_objective_experiment = MultiObjectiveSimpleExperiment(
        ax_search_space=ax_cs,
        cs_search_space=cs,
        optimization_config=optimization_config,
        output_path=output_path,

        # Time Limits
        tae_limit=tae_limit,
        wallclock_limit_in_s=wallclock_limit_in_s,
        estimated_cost_limit_in_s=estimated_cost_limit_in_s,
        is_surrogate=is_surrogate,

        name=name,
        eval_function=wrapped_benchmark.eval_function,
        status_quo=None,
        properties=None,
        extra_metrics=extra_metrics,
    )
    return multi_objective_experiment
