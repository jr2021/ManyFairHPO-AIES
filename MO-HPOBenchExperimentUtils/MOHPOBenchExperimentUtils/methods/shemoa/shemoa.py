import math

import numpy as np
from ax import Experiment
from .member import Member
from .member import Mutation
from .member import ParentSelection
from .member import Recombination
from MOHPOBenchExperimentUtils import nDS_index, contributionsHV3D, computeHV, nDS
from loguru import logger
import sys
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

class SHEMOA:
    """
    Succesive Halving Evolutionary MultiObjective Algorithm
    :param population_size: int
    :param mutation_type: hyperparameter to set mutation strategy
    :param recombination_type: hyperparameter to set recombination strategy
    :param sigma: conditional hyperparameter dependent on mutation_type GAUSSIAN
    :param recom_proba: conditional hyperparameter dependent on recombination_type UNIFORM
    :param selection_type: hyperparameter to set selection strategy
    :param total_number_of_function_evaluations: maximum allowed function evaluations
    :param children_per_step: how many children to produce per step
    :param fraction_mutation: balance between sexual and asexual reproduction
    """

    def __init__(
            self,
            search_space,
            experiment: Experiment,
            population_size: int = 10,
            budget_min: int = 5,
            budget_max: int = 50,
            eta: int = 2,
            init_time: float = 0.0,
            mutation_type: Mutation = Mutation.UNIFORM,
            recombination_type:
            Recombination = Recombination.UNIFORM,
            sigma: float = 1.,
            recom_proba: float = 0.5,
            selection_type: ParentSelection = ParentSelection.TOURNAMENT,
            total_number_of_function_evaluations: int = 200,
            children_per_step: int = 1,
            fraction_mutation: float = .5,
            use_hv_to_sort=True,
            adapt_cfg_function=None
    ):
        assert 0 <= fraction_mutation <= 1
        assert 0 < children_per_step
        assert 0 < total_number_of_function_evaluations
        assert 0 < sigma
        assert 0 < population_size

        self.current_budget = 0  # first budget
        self.init_time = init_time
        self.experiment = experiment
        self.sort_by_hv = use_hv_to_sort
        # Compute succesive halving values
        budgets, evals = get_budgets(
            budget_min,
            budget_max,
            eta,
            total_number_of_function_evaluations,
            population_size
        )


        # Initialize population
        self.population = [
            Member(
                search_space,
                budgets[0],
                mutation_type,
                recombination_type,
                'dummy.txt',
                sigma,
                recom_proba,
                experiment=self.experiment,
                adapt_cfg_function=adapt_cfg_function
            ) for _ in range(population_size)
        ]

        # This function sorts the population using HV if 'use_hv_to_sort' is True
        # If this flag is set flag the sorting method of original implementation is used which is
        # experiment specific and ideally should be only to check reproducibility

        self.sort_population()
        self.pop_size = population_size
        self.selection = selection_type
        self.max_func_evals = total_number_of_function_evaluations
        self._func_evals = population_size
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        # will store the optimization trajectory and lets you easily observe how often
        # a new best member was generated
        self.trajectory = [self.population[0]]
        # list of different budgets
        self.budgets = budgets
        # list of how many function evaluations to do per budget
        self.evals = evals

    def select_parents(self):
        """
        Method that implements all selection mechanism.
        For ease of computation we assume that the population members are sorted according to their fitness
        :return: list of ids of selected parents.
        """
        parent_ids = []
        if self.selection == ParentSelection.NEUTRAL:
            parent_ids = np.random.choice(self.pop_size, self.num_children)
        elif self.selection == ParentSelection.FITNESS:
            p = np.array([x.fitness for x in self.population])
            p = (p.max() - p) + 0.0001
            p = p / p.sum()
            parent_ids = np.random.choice(self.pop_size, self.num_children, p=p)

        elif self.selection == ParentSelection.TOURNAMENT:
            k = 3
            parent_ids = [np.random.choice(self.pop_size, min(k, self.pop_size), replace=False).min()
                          for i in range(self.num_children)]
        else:
            raise NotImplementedError
        return parent_ids

    def remove_member(self, fitness):
        # BE CAREFUL: fitness must be a list
        for m in self.population:
            if list(m.fitness) == fitness:
                self.population.remove(m)
                break
        else:
            raise Warning("remove_member did not found the member to remove")
        return m.id


    def _get_paretos(self, population_fitness):
        index_list = np.array(list(range(len(population_fitness))))
        fitness = np.array([np.array(x) for x in population_fitness])
        return nDS_index(np.array(fitness), index_list)

    def _select_best_hv(self, population_fitness):
        fronts, index_return_list = self._get_paretos(population_fitness)
        sorted = []
        for idx, front in enumerate(fronts):
            hv = np.array(contributionsHV3D(front))
            sort_index = np.argsort(-1 * hv)
            indexes = index_return_list[idx]
            sorted.extend(indexes[sort_index])
        return sorted



    def step(self) -> float:
        """
        Performs one step of parent selection -> offspring creation -> survival selection
        :return: average population fitness
        """
        # Step 2: Parent selection
        parent_ids = self.select_parents()
        children = []
        for pid in parent_ids:
            if np.random.uniform() < self.frac_mutants:
                children.append(self.population[pid].mutate())
            else:
                children.append(self.population[pid].recombine(np.random.choice(self.population)))
            self._func_evals += 1

        # Step 4: Survival selection
        # (\mu + \lambda)-selection i.e. combine offspring and parents in one sorted list, keep the #pop_size best
        self.population.extend(children)
        costs = np.array([x.fitness for x in self.population])
        fronts = nDS(costs)

        if len(fronts[-1]) == 1:
            r_member = fronts[-1][0].tolist()
            r_id = self.remove_member(r_member)
        else:
            # sort the front for hv calculation
            sfront_indexes = np.argsort(fronts[-1][:, 1])
            sfront = fronts[-1][sfront_indexes, :]
            #This is modified from original implementation to calculate reference point considering worse points
            hv = computeHV(sfront)
            min_hvc = hv
            min_point = sfront[0]
            sfront = sfront.tolist()
            for point in sfront:
                front_wout = sfront.copy()
                front_wout.remove(point)
                hvc = hv - computeHV(front_wout)
                if hvc < min_hvc:
                    min_hvc = hvc
                    min_point = point

            r_id = self.remove_member(min_point)
        self.sort_population()
        self.trajectory.append(self.population[0])



    def optimize(self):
        """
        Simple optimization loop that stops after a predetermined number of function evaluations
        :return:
        """
        step = 1
        for b in range(len(self.budgets)):
            if b > 0:
                self.current_budget = b
                # Change budget for all members in population (train them for bigger budget)
                for m in self.population:
                    m.budget = self.budgets[b]
                # Re-order the population given the new fitness with bigger budget
                self.sort_population()
            for e in range(self.evals[b]):
                self.step()
                step += 1

        # Calculate pareto front of population
        costs = np.array([x.fitness for x in self.population])
        fronts = nDS(costs)
        pareto = []
        pareto_front = fronts[0].tolist()
        for m in self.population:
            if list(m.fitness) in pareto_front:
                pareto.append(m)
        return pareto

    def sort_population(self):
        if self.sort_by_hv:
            costs = np.array([x.fitness for x in self.population])
            sorted_index = self._select_best_hv(costs)
            sorted_pop = [self.population[i] for i in sorted_index]
            self.population = sorted_pop
        else:
            self.population.sort(key=lambda x: (15 - x.fitness[0]) * x.fitness[1])


def get_budgets(bmin, bmax, eta, max_evals, pop_size):
    # Size of all budgets
    budgets = []
    b = bmax
    while b > bmin:
        budgets.append(b)
        b = math.ceil(b / eta)
    # Number of function evaluations to do per budget
    evals = []
    min_evals = math.ceil((max_evals - pop_size) / sum([eta ** i for i in range(len(budgets))]))
    for _ in range(len(budgets)):
        evals.append(min_evals)
        min_evals = eta * min_evals

    logger.debug(f"budget bracket:{np.array(budgets)}")
    logger.debug(f'evaluation brackets:{np.array(evals)}')

    return np.flip(np.array(budgets)), np.flip(np.array(evals))
