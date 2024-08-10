from typing import Dict, List, Union

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter
from loguru import logger
from pymoo.core.problem import ElementwiseProblem

from MOHPOBenchExperimentUtils.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
from MOHPOBenchExperimentUtils.methods.base_optimizer import Optimizer
from MOHPOBenchExperimentUtils.methods.pymoo_.custom_termination import CustomMultiObjectiveSpaceToleranceTermination
from pymoo.util.termination.no_termination import NoTermination


class HPOBenchProblem(ElementwiseProblem):

    def __init__(self, experiment: MultiObjectiveSimpleExperiment,
                 lower_limits: List, upper_limits: List, constant_values: List[CS.Constant],
                 fidelity_settings: Dict, num_hp=0, num_objectives=0, num_constraints=0):

        self.lower_limits = np.array(lower_limits)
        self.upper_limits = np.array(upper_limits)
        self.constant_values = constant_values
        self.constant_values = {hp.name: hp.value for hp in self.constant_values}
        self.experiment = experiment

        self.max_fidelity: Dict[str, Union[int, float]] = \
            {entry['name']: entry['limits'][1] for entry in fidelity_settings}

        super().__init__(n_var=num_hp, n_obj=num_objectives, n_constr=num_constraints,
                         xl=self.lower_limits, xu=self.upper_limits)

    def _evaluate(self, x, out, *args, **kwargs):

        assert (len(x) + len(self.constant_values)) == len(self.experiment.cs_search_space)
        # The pymoo search space contains only non-constant parameters. We bypass them in this step to
        # the underlying objectve function.
        non_constant_parameters = [
            hp for hp in self.experiment.cs_search_space.get_hyperparameters()
            if not isinstance(hp, CS.Constant)
        ]
        configuration = {hp.name: value for hp, value in zip(non_constant_parameters, x)}
        configuration = {**configuration, **self.constant_values}

        data, metrics = self.experiment.eval_configuration(configuration, self.max_fidelity)

        metric_val = [float(data.df[data.df['metric_name'] == obj]['mean']) for obj in metrics]
        logger.debug("metric val:{}", metric_val)
        out['F'] = metric_val


class Pymoo_Optimizer(Optimizer):
    def __init__(self,
                 optimizer_settings: Dict, benchmark_settings: Dict,
                 configspace: CS.ConfigurationSpace,
                 wallclock_limit_in_s: Union[int, float]):

        self.algorithm = None
        super(Pymoo_Optimizer, self).__init__(optimizer_settings, benchmark_settings, configspace)

        # self.default_termination = CustomMultiObjectiveSpaceToleranceTermination(
        #     tol=10e-10,  # tolerance in the objective space on average
        #     nth_gen=5,  # Defines whenever the termination criterion is calculated
        #     n_last=30,  # To make the criterion more robust, we consider the last n generations and take the maximum.
        #     n_max_gen=10000000000000000,  # Set to  almost inf
        #     n_max_evals=10000000000000000,  # Set to almost inf,
        #     max_time=10000000000000000,  # Set to almost inf,
        # )
        self.default_termination = NoTermination()

    def get_pymoo_cs(self):
        hps = self.cs.get_hyperparameters()

        parameter_types = []
        lower_limits = []
        upper_limits = []
        constant_values = []

        for hp in hps:
            if isinstance(hp, IntegerHyperparameter):
                parameter_types.append('int')
            elif isinstance(hp, FloatHyperparameter):
                parameter_types.append('real')
            elif isinstance(hp, CS.Constant):
                constant_values.append(hp)
                continue
            else:
                raise ValueError(f'Currently Not Supported HP Type {type(hp)}')
            lower_limits.append(hp.lower)
            upper_limits.append(hp.upper)

        return parameter_types, lower_limits, upper_limits, constant_values

    def init(self):
        raise NotImplementedError()

    def setup(self, problem: HPOBenchProblem, seed: int):
        raise NotImplementedError()

    def run(self):
        generation = 0
        while self.algorithm.has_next():
            logger.info(f'Generation {generation}')
            self.algorithm.next()
            generation += 1