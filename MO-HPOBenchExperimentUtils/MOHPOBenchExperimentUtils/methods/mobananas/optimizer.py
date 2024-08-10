import ConfigSpace as CS
from MOHPOBenchExperimentUtils.methods.base_optimizer import Optimizer

from typing import Dict, Union
from MOHPOBenchExperimentUtils.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
from loguru import logger

from baselines.methods.mobananas.mobananas import BANANAS as MO_BANANAS
from baselines.methods.mobananas.moshbananas import BANANAS as MOSH_BANANAS
from baselines.methods.mobananas.neural_predictor import Neural_Predictor
from baselines.methods.mobananas.member import Mutation

"""
Please download the corresponding code from: 
https://github.com/PhMueller/MOExp-BagOfBaselines

and install with 

``` python -e . ```

"""


class MOBananasBaseOptimizer(Optimizer):
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configspace: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(MOBananasBaseOptimizer, self).__init__(optimizer_settings, benchmark_settings, configspace, **kwargs)
        self.experiment = None
        self.algorithm = None
        self.neural_predictor = None

        assert len(self.benchmark_settings['fidelity']) == 1, \
            f'This optimizer does not support Multi-Multi-Fidelity. However, you specified a setting with ' \
            f'{len(self.benchmark_settings["fidelity"])} fidelities.'
        self.first_fidelity = self.benchmark_settings['fidelity'][0]

    def init(self, experiment: MultiObjectiveSimpleExperiment, **kwargs):
        self.experiment = experiment

    def setup(self, seed: int = 0, **kwargs):
        raise NotImplementedError()

    def run(self, **kwargs):
        logger.info('Start Optimization Run')
        self.algorithm.steps()
        logger.info(f'Finished Optimization Run')


class MOBananasOptimizer(MOBananasBaseOptimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configspace: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(MOBananasOptimizer, self).__init__(optimizer_settings, benchmark_settings, configspace, **kwargs)

    def setup(self, seed: int = 0, **kwargs):
        neural_predictor_settings = self.optimizer_settings['neural_predictor_settings']
        algorithm_settings = self.optimizer_settings['algorithm_settings']

        num_input_parameters = len(
             [hp for hp in self.experiment.cs_search_space.get_hyperparameters() if not isinstance(hp, CS.Constant)]
        )
        self.neural_predictor = Neural_Predictor(
            num_input_parameters=num_input_parameters,
            num_objectives=len(self.benchmark_settings['metrics']['target']),
            **neural_predictor_settings
        )

        self.algorithm = MO_BANANAS(
            neural_predictor=self.neural_predictor,
            experiment=self.experiment,
            search_space=self.experiment.cs_search_space,
            mutation_type=Mutation.GAUSSIAN,
            budget=self.first_fidelity,
            function_evaluations=self.experiment.tae_limit,
            seed=seed,
            **algorithm_settings,
        )


class MOSHBananasOptimizer(MOBananasBaseOptimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configspace: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(MOSHBananasOptimizer, self).__init__(optimizer_settings, benchmark_settings, configspace, **kwargs)

    def setup(self, seed: int = 0, **kwargs):
        neural_predictor_settings = self.optimizer_settings['neural_predictor_settings']
        algorithm_settings = self.optimizer_settings['algorithm_settings']

        num_input_parameters = len(
            [hp for hp in self.experiment.cs_search_space.get_hyperparameters() if not isinstance(hp, CS.Constant)]
        )

        self.neural_predictor = Neural_Predictor(
            num_input_parameters=num_input_parameters,
            num_objectives=len(self.benchmark_settings['metrics']['target']),
            **neural_predictor_settings
        )

        self.algorithm = MOSH_BANANAS(
            neural_predictor=self.neural_predictor,
            experiment=self.experiment,
            search_space=self.experiment.cs_search_space,
            mutation_type=Mutation.GAUSSIAN,
            budget=self.first_fidelity,
            function_evaluations=self.experiment.tae_limit,
            seed=seed,
            **algorithm_settings,
        )

