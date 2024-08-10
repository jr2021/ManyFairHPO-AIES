from typing import Dict, Union

import ConfigSpace as CS
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, MixedVariableMutation

from MOHPOBenchExperimentUtils.methods.pymoo_.pymoo_base import Pymoo_Optimizer, HPOBenchProblem


class Pymoo_NSGA3(Pymoo_Optimizer):

    def __init__(self, optimizer_settings: Dict, benchmark_settings: Dict,
                 configspace: CS.ConfigurationSpace, wallclock_limit_in_s: Union[int, float]):

        super(Pymoo_NSGA3, self).__init__(
            optimizer_settings=optimizer_settings,
            benchmark_settings=benchmark_settings,
            configspace=configspace,
            wallclock_limit_in_s=wallclock_limit_in_s
        )

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

        # TODO: How to set the number of partions?
        #       Are pymoo_ optimizers expecting [0, 1] objectives? Do the ref points play a role here?
        ref_directions = get_reference_directions(
            'das-dennis',
            n_partitions=self.optimizer_settings['ref_dir_n_partitions'],
            n_dim=len(self.benchmark_settings['metrics']['target'])
        )

        # TODO: Define a good population size
        self.algorithm = NSGA3(
            ref_dirs=ref_directions,
            pop_size=self.optimizer_settings['pop_size'],
            eliminate_duplicates=self.optimizer_settings['eliminate_duplicates'],
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )

    def setup(self, problem: HPOBenchProblem, seed: int):
        self.algorithm.setup(
            problem=problem, seed=seed,
            # Maybe, we want to overwrite some of the following parameters later.
            termination=self.default_termination, callback=None, display=None, evaluator=None
        )
