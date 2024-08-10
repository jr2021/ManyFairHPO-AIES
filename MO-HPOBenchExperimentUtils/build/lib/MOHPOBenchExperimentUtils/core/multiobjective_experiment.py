import random
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import time

import ConfigSpace as CS
from loguru import logger
from ConfigSpace import ConfigurationSpace as CSConfigurationSpace
from ax import Metric, Experiment, SimpleExperiment, \
    OptimizationConfig, GeneratorRun, Arm, SearchSpace as AxSearchSpace

from ax.core.simple_experiment import TEvaluationFunction
from ax.core.simple_experiment import unimplemented_evaluation_function

from scripts.log_outputs import save_output


class MultiObjectiveSimpleExperiment(SimpleExperiment):

    def __init__(
        self,
        ax_search_space: AxSearchSpace,
        cs_search_space: CSConfigurationSpace,
        optimization_config: OptimizationConfig,
        output_path: Union[str, Path],
        tae_limit: Union[int, None],
        wallclock_limit_in_s: Union[int, float],
        estimated_cost_limit_in_s: Union[int, float],
        is_surrogate: bool = False,
        name: Optional[str] = None,
        eval_function: TEvaluationFunction = unimplemented_evaluation_function,
        status_quo: Optional[Arm] = None,
        properties: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[List[Metric]] = None,
    ):
        super(MultiObjectiveSimpleExperiment, self).__init__(
            search_space=ax_search_space,
            name=name,
            evaluation_function=eval_function,
            status_quo=status_quo,
            properties=properties
        )

        self.optimization_config = optimization_config
        self.cs_search_space = cs_search_space
        self.output_path = Path(output_path)

        self.tae_limit = tae_limit
        self.wallclock_limit_in_s = wallclock_limit_in_s
        self.estimated_cost_limit_in_s = estimated_cost_limit_in_s
        self.is_surrogate = is_surrogate

        self.initial_time = time.time()
        self.num_configs_evaluated = 0
        self.accumulated_surrogate_cost = 0
        self.time_for_saving = 0
        self.used_wallclock_time = 0
        self.used_total_cost = 0
        self.extra_metrics = extra_metrics

        if extra_metrics is not None:
            for metric in extra_metrics:
                Experiment.add_tracking_metric(self, metric)

        if self.tae_limit == -1:
            self.tae_limit = len(self.cs_search_space.get_hyperparameters()) * 1000
            logger.info(
                f'The Target algorithm execution limit is -1.'
                f' Set it to its default value => 1000 * #hps = {self.tae_limit}'
            )

    def start_timer(self):
        self.initial_time = time.time()
        self.used_wallclock_time = 0
        self.used_total_cost = 0
        self.accumulated_surrogate_cost = 0

    def eval_configuration(self, configuration: Dict, fidelity: Dict = None):

        if isinstance(configuration, CS.Configuration):
            configuration = configuration.get_dictionary()

        # Check that the wallclock time limit is not reached
        self.used_wallclock_time = time.time() - self.initial_time

        if self.used_wallclock_time >= self.wallclock_limit_in_s:
            logger.warning(f'Wallclock Time Limit Reached')
            raise TimeoutError(f'Wallclock Time Limit Reached: {self.wallclock_limit_in_s}')

        # If we evaluate a surrogate model then we also care about the predicted costs
        self.used_total_cost = self.used_wallclock_time + self.accumulated_surrogate_cost
        if self.is_surrogate and (self.used_total_cost >= self.estimated_cost_limit_in_s):
            logger.warning(f'Surrogate Costs Limit Reached')
            raise TimeoutError(f'Total Time Limit Reached: {self.estimated_cost_limit_in_s}')

        # By default, the tae limit is not active. If it is set to -1, 1000 * #hps is used.
        if self.tae_limit is not None and self.num_configs_evaluated > self.tae_limit:
            logger.warning('Target Execution limit is reached.')
            raise TimeoutError(f'Total Number of Target Executions Limit is reached. {self.num_configs_evaluated}')

        # --------------------------- PREPARE CONFIG AND FIDELITY --------------------------------
        if fidelity is not None:
            configuration = {**configuration, **fidelity}

        # We have to add missing parameters (even if they are not active), since ax will raise an
        # error otherwise. HPObench will remove them before running the configuration, therefore
        # we should be okay with this solution. (more or less)
        for key in self.cs_search_space.get_hyperparameter_names():
            if key not in configuration:
                missing_hp = self.cs_search_space.get_hyperparameter(key)
                configuration[key] = missing_hp.default_value

        # Add a random configuration id to the configuration to fix an Ax related problem.
        configuration['id'] = str(random.getrandbits(128))

        trial = self.new_trial(GeneratorRun([
            Arm(configuration, name=str(self.num_configs_evaluated))
        ]))
        self.num_configs_evaluated += 1

        # --------------------------- EVALUATE TRIAL --------------------------------
        data = self.eval_trial(trial)

        # --------------------------- UPDATE TIMER --------------------------------
        if self.is_surrogate:
            returned_costs = data.df.loc[data.df.metric_name == 'cost', 'mean']
            self.accumulated_surrogate_cost += returned_costs.item()

        # --------------------------- POST PROCESS --------------------------------
        objectives = self.optimization_config.objective.metrics
        metric_names = [obj.name for obj in objectives]

        eval_interval = 100 if self.is_surrogate else 10
        if (self.num_configs_evaluated % eval_interval) == 0:
            remaining_time = self.wallclock_limit_in_s - self.used_wallclock_time
            logger.info(f'WallClockTime left: {remaining_time:10.4f}s ({remaining_time/3600:.4f}h)')
            if self.is_surrogate:
                remaining_time = self.estimated_cost_limit_in_s - self.used_total_cost
                logger.info(f'EstimatedTime left: {remaining_time:10.4f}s ({remaining_time/3600:.4f}h)')
            if self.tae_limit is not None:
                logger.info(f'Number of TAE: {self.num_configs_evaluated:10d}|{self.tae_limit}')

        save_interval = 10000 if self.is_surrogate else 10
        if (self.num_configs_evaluated % save_interval) == 0:
            # Saving intermediate results is pretty expensive
            t = time.time()
            save_output(self, self.output_path, finished=False, surrogate=self.is_surrogate)
            time_for_saving = time.time() - t
            logger.info(f'Saved Experiment to Pickle took {time_for_saving:.2f}s')

        return data, metric_names
