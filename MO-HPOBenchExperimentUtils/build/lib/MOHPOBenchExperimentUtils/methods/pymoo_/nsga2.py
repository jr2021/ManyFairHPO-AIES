from typing import Dict, Union

import ConfigSpace as CS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, MixedVariableMutation
from loguru import logger
from pymoo.util.display import Display
from pymoo.util.termination.max_gen import MaximumGenerationTermination

from MOHPOBenchExperimentUtils.methods.pymoo_.pymoo_base import Pymoo_Optimizer, HPOBenchProblem

class Pymoo_NSGA2(Pymoo_Optimizer):

    def __init__(self, optimizer_settings: Dict, benchmark_settings: Dict,
                 configspace: CS.ConfigurationSpace, wallclock_limit_in_s: Union[int, float]):

        super(Pymoo_NSGA2, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configspace=configspace,
            wallclock_limit_in_s=wallclock_limit_in_s
        )
        self.max_time = wallclock_limit_in_s
        self.pop_size = optimizer_settings['pop_size']
        self.n_max_gen = optimizer_settings['n_max_gen']


    def init(self):
        parameter_types, _, _, _ = self.get_pymoo_cs()

        sampling = MixedVariableSampling(parameter_types, {
            "real": get_sampling("real_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(parameter_types, {
            "real": get_crossover("real_sbx", prob=1.0, eta=30),
            "int": get_crossover("int_sbx", prob=1.0, eta=30)
        })

        mutation = MixedVariableMutation(parameter_types, {
            "real": get_mutation("real_pm", eta=20, prob=None),
            "int": get_mutation("int_pm", eta=20, prob=None)
        })

        self.algorithm = NSGA2(
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            pop_size=self.pop_size
        )

        self.termination = MaximumGenerationTermination(n_max_gen=self.n_max_gen)

    def setup(self, problem: HPOBenchProblem, seed: int):    
        self.algorithm.setup(
            problem=problem, seed=seed,
            # Maybe, we want to overwrite some of the following parameters later.
            termination=self.termination, callback=None, evaluator=None
        )
