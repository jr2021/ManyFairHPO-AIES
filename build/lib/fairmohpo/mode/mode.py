from copy import deepcopy
from typing import Callable, Dict, List, NamedTuple, Optional

import ConfigSpace as CS

import numpy as np

from fairmohpo.mode.utils import (
    _boundary_check,
    _denormalize_vector,
    _initial_sampling,
)
from fairmohpo.mode.components import CrossOverFuncs, MutationFuncs, SelectionFuncs


class _ComponentParams(NamedTuple):
    boundary_fix_type: str
    mutation_factor: float
    crossover_prob: float
    mutation_strategy: str
    crossover_strategy: str
    selection_strategy: str


class DifferentialEvolutionPopulation:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        pop_size: int,
        obj_func: Callable,
        params: _ComponentParams,
        result_keys: List[str],
        seed: Optional[int],
    ):
        self._params = params
        self._pop_size = pop_size
        self._population = _initial_sampling(config_space=config_space, pop_size=pop_size)
        self._fitness = np.full((pop_size, len(result_keys)), np.inf)
        self._obj_func = obj_func
        self._result_keys = result_keys[:]
        self._rng = np.random.RandomState(seed)

    def _sample(self, n_samples: int) -> np.ndarray:
        selection = self._rng.choice(np.arange(self._pop_size), n_samples, replace=False)
        return self._population[selection]

    def _mutation(
        self,
        current: Optional[np.ndarray] = None,
        best: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Performs DE mutation"""
        mutation_strategy, mutation_factor = self._params.mutation_strategy, self._params.mutation_factor
        multiobj = len(self._result_keys) > 1
        return MutationFuncs.mutation_func(
            cands=self._sample(n_samples=MutationFuncs.cands_size[mutation_strategy]),
            mutation_strategy=mutation_strategy,
            mutation_factor=mutation_factor,
            best=best if multiobj or best is not None else self._population[np.argmin(self._fitness)],
            current=current,
        )

    def _crossover(self, parent: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Performs DE crossover"""
        return CrossOverFuncs.crossover_func(
            crossover_strategy=self._params.crossover_strategy,
            parent=parent,
            mutant=mutant,
            crossover_prob=self._params.crossover_prob,
            rng=self._rng,
        )

    def _selection(self, children: np.ndarray) -> None:
        """Carries out a parent-offspring competition given a set of trial population"""
        selection_args = dict(strategy=self._params.selection_strategy, child_idx=-1)
        for parent_idx, child in enumerate(children):
            results = self._obj_func(child)
            fitness = np.asarray([results[key] for key in self._result_keys])
            costs = np.vstack([self._fitness, fitness])
            selection_args.update(costs=costs, parent_idx=parent_idx)

            if SelectionFuncs.selection(**selection_args):
                self._population[parent_idx] = child
                self._fitness[parent_idx] = fitness

    def evolve_generation(self, best: Optional[np.ndarray] = None) -> None:
        _children: List[np.ndarray] = []
        for parent in self._population:
            mutant = self._mutation(current=parent, best=best)
            child = self._crossover(parent=parent, mutant=mutant)
            child = _boundary_check(child, fix_type=self._params.boundary_fix_type, rng=self._rng)
            _children.append(child)

        self._selection(np.asarray(_children))


class DEOptimizer:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        obj_func: Callable,
        result_keys: List[str],
        ref_point: int = 1,
        pop_size: int = 20,
        mutation_factor: float = 0.5,
        crossover_prob: float = 0.5,
        mutation_strategy: str = "rand1",
        crossover_strategy: str = "bin",
        boundary_fix_type: str = "random",
        selection_strategy: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        config_space.seed(seed)  # reproducibility
        default_selection_strategy = "multi" if len(result_keys) > 1 else "single"
        params = _ComponentParams(
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy,
            boundary_fix_type=boundary_fix_type,
            selection_strategy=default_selection_strategy if selection_strategy is None else selection_strategy,
        )
        self._pop = DifferentialEvolutionPopulation(
            config_space=config_space,
            pop_size=pop_size,
            obj_func=self.f_objective,
            params=params,
            result_keys=result_keys,
            seed=seed,
        )
        self._result_keys = result_keys[:]
        self._obj_func = obj_func
        self._ref_point = ref_point
        self._config_space = config_space
        self._all_keys = ['f1', 'ddsp', 'deop', 'deod', 'genr']
        self._observations: Dict[str, List] = {hp_name: [] for hp_name in config_space}
        self._observations.update({metric_name: [] for metric_name in self._all_keys})
        self._observations.update({f"{metric_name}_test": [] for metric_name in self._all_keys})

    def f_objective(self, x: np.ndarray) -> Dict[str, float]:
        config = _denormalize_vector(config_space=self._config_space, vector=x)
        for hp_name in self._config_space:
            self._observations[hp_name].append(config[hp_name])

        results, test_results, meta_results = self._obj_func(config) 

        for metric_name, val in results.items():
            self._observations[metric_name].append(val)

        for metric_name, val in test_results.items(): 
            self._observations[f"{metric_name}_test"].append(val)

        for metric_name, val in meta_results.items():
            self._observations[metric_name].append(val)

        if any(key not in results for key in self._result_keys):
            raise KeyError(f"The results must include {self._result_keys}, but got {results}")

        return results

    def run(self, generations) -> None:
        for i in range(generations):
            print("######################### generation", i, "#########################")
            self._pop.evolve_generation()
            
    @property
    def population(self) -> np.ndarray:
        return deepcopy(self._pop._population)

    @property
    def fitness(self) -> np.ndarray:
        return deepcopy(self._pop._fitness)

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {name: vals.copy() for name, vals in self._observations.items()}