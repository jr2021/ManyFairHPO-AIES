import ConfigSpace as CS
from MOHPOBenchExperimentUtils.methods.base_optimizer import Optimizer

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.optimizer.multi_objective.parego import ParEGO


from typing import Dict, Union
from MOHPOBenchExperimentUtils.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
from loguru import logger
from copy import deepcopy


MAP_MULTI_OBJECTIVE_ALGORITHMS = {'ParEGO': ParEGO}
MAP_SMAC_FACADE = {'SMAC4HPO': SMAC4HPO, 'SMAC4BB': SMAC4BB}


class SMACOptimizer(Optimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configspace: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(SMACOptimizer, self).__init__(optimizer_settings, benchmark_settings, configspace, **kwargs)

        self.algorithm = None
        self.experiment = None
        self._objective_function = None
        self.scenario = None

    def init(self, experiment: MultiObjectiveSimpleExperiment, **kwargs):

        self.experiment = experiment

        max_fidelity: Dict[str, Union[int, float]] = \
            {entry['name']: entry['limits'][1] for entry in self.benchmark_settings['fidelity']}

        def _objective_function(cfg: Dict):
            data, metric_names = self.experiment.eval_configuration(configuration=cfg, fidelity=max_fidelity)
            result_dict = {obj: float(data.df[data.df['metric_name'] == obj]['mean']) for obj in metric_names}
            logger.debug(f'Returned objectives: {result_dict}')
            return result_dict

        self._objective_function = _objective_function

        target_names = [
            target['name'] for target in self.benchmark_settings['metrics']['target']
        ]

        target_upper_limits = [
            (1 if target['lower_is_better'] else -1) * target['threshold']
            for target in self.benchmark_settings['metrics']['target']
        ]

        self.scenario = Scenario(
            {
                "run_obj": self.optimizer_settings['run_objective'],
                "runcount-limit": self.optimizer_settings['run_count_limit'],
                "cs": self.cs,
                "deterministic": self.optimizer_settings['deterministic'],
                "multi_objectives": target_names,

                # You can define individual crash costs for each objective
                # "cost_for_crash": [float(MAXINT), float(MAXINT)],
                "cost_for_crash": target_upper_limits,
            }
        )

    def setup(self, seed: int = 0, **kwargs):

        algorithm_parameters = deepcopy(self.optimizer_settings['algorithm_options'])
        algorithm_name = algorithm_parameters.pop('smac_facade')
        algorithm_type = MAP_SMAC_FACADE[algorithm_name]

        if algorithm_name == 'SMAC4HPO':
            mo_algo_name = algorithm_parameters['multi_objective_algorithm']
            algorithm_parameters['multi_objective_algorithm'] = MAP_MULTI_OBJECTIVE_ALGORITHMS[mo_algo_name]

        self.algorithm = algorithm_type(
            scenario=self.scenario,
            rng=seed,
            tae_runner=self._objective_function,
            **algorithm_parameters,
        )

    def run(self, **kwargs):

        logger.info('Start Optimization Run')
        incumbent = self.algorithm.optimize()
        logger.info(f'Finished Optimization Run with incumbent {incumbent}')
