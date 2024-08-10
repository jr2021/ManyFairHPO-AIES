from typing import Optional

from fast_pareto import nondominated_rank

import numpy as np


class SelectionFuncs:
    strategy_choices = ["single", "multi"]

    # selection -- competition between parent[i] -- child[i]
    @staticmethod
    def _single(
        costs: np.ndarray,
        parent_idx: int,
        child_idx: int,
    ) -> bool:
        # equality is important for landscape exploration
        parent_fitness = costs[parent_idx][0]
        child_fitness = costs[child_idx][0]
        assert costs.shape[-1] == 1
        assert len(costs.shape) == 2
        return child_fitness <= parent_fitness

    @staticmethod
    def _multi(
        costs: np.ndarray,
        parent_idx: int,
        child_idx: int,
    ) -> bool:
        nd_rank = nondominated_rank(costs)
        # Use average rank tie-break for the same non-dominated ranks
        return nd_rank[child_idx] <= nd_rank[parent_idx]

    @classmethod
    def selection(
        cls,
        strategy: str,
        costs: np.ndarray,
        parent_idx: int,
        child_idx: int,
    ) -> bool:
        """Choose parent -> False, choose child -> True"""
        if strategy not in cls.strategy_choices:
            raise ValueError(f"strategy must be in {cls.strategy_choices}, but got {strategy}")

        assert isinstance(costs[parent_idx], np.ndarray)
        assert isinstance(costs[child_idx], np.ndarray)
        return getattr(cls, f"_{strategy}")(
            costs=costs,
            parent_idx=parent_idx,
            child_idx=child_idx,
        )


class MutationFuncs:
    cands_size = {
        "rand1": 3,
        "rand2": 5,
        "currenttobest1": 2,
        "rand2dir": 3,
        "best1": 2,
        "best2": 4,
        "randtobest1": 3,
    }

    @classmethod
    def get_min_pop_size(cls, mutation_strategy: str) -> int:
        return cls.cands_size.get(mutation_strategy, 1)

    @staticmethod
    def _mutation_rand1(cands: np.ndarray, mutation_factor: float, **kwargs) -> np.ndarray:
        """Performs the 'rand1' type of DE mutation"""
        diff = cands[1] - cands[2]
        mutant = cands[0] + mutation_factor * diff
        return mutant

    @staticmethod
    def _mutation_rand2(cands: np.ndarray, mutation_factor: float, **kwargs) -> np.ndarray:
        """Performs the 'rand2' type of DE mutation"""
        diff1 = cands[1] - cands[2]
        diff2 = cands[3] - cands[4]
        mutant = cands[0] + mutation_factor * diff1 + mutation_factor * diff2
        return mutant

    @staticmethod
    def _mutation_rand2dir(cands: np.ndarray, mutation_factor: float, **kwargs) -> np.ndarray:
        diff = cands[0] - cands[1] - cands[2]
        mutant = cands[0] + mutation_factor * diff / 2
        return mutant

    @staticmethod
    def _mutation_best1(cands: np.ndarray, mutation_factor: float, best: np.ndarray, **kwargs) -> np.ndarray:
        """Performs the 'rand1' type of DE mutation"""
        diff = cands[0] - cands[1]
        mutant = best + mutation_factor * diff
        return mutant

    @staticmethod
    def _mutation_best2(cands: np.ndarray, mutation_factor: float, best: np.ndarray, **kwargs) -> np.ndarray:
        """Performs the 'rand2' type of DE mutation"""
        diff1 = cands[0] - cands[1]
        diff2 = cands[2] - cands[3]
        mutant = best + mutation_factor * diff1 + mutation_factor * diff2
        return mutant

    @staticmethod
    def _mutation_currenttobest1(
        current: np.ndarray, best: np.ndarray, cands: np.ndarray, mutation_factor: float
    ) -> np.ndarray:
        diff1 = best - current
        diff2 = cands[0] - cands[1]
        mutant = current + mutation_factor * diff1 + mutation_factor * diff2
        return mutant

    @staticmethod
    def _mutation_randtobest1(best: np.ndarray, cands: np.ndarray, mutation_factor: float, **kwargs) -> np.ndarray:
        diff1 = best - cands[0]
        diff2 = cands[1] - cands[2]
        mutant = cands[0] + mutation_factor * diff1 + mutation_factor * diff2
        return mutant

    @classmethod
    def mutation_func(
        cls,
        cands: np.ndarray,
        mutation_strategy: str,
        mutation_factor: float,
        current: Optional[np.ndarray] = None,
        best: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        assert len(cands.shape) == 2
        assert len(cands) == cls.cands_size[mutation_strategy]
        mutation_args = dict(cands=cands, mutation_factor=mutation_factor, best=best, current=current)
        strategy_choices = list(cls.cands_size.keys())

        if mutation_strategy not in strategy_choices:
            raise ValueError(f"mutation_strategy must be in {strategy_choices}, but got {mutation_strategy}")

        if "best" in strategy_choices:
            assert best is not None
        if "current" in strategy_choices:
            assert current is not None

        return getattr(cls, f"_mutation_{mutation_strategy}")(**mutation_args)


class CrossOverFuncs:
    strategy_choices = ["bin", "exp"]

    @staticmethod
    def _crossover_bin(
        parent: np.ndarray, mutant: np.ndarray, crossover_prob: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """Performs the binomial crossover of DE"""
        dim = parent.size
        cross_points = rng.rand(dim) < crossover_prob
        if not np.any(cross_points):
            cross_points[rng.randint(0, dim)] = True

        child = np.where(cross_points, mutant, parent)
        return child

    @staticmethod
    def _crossover_exp(
        parent: np.ndarray, mutant: np.ndarray, crossover_prob: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """Performs the exponential crossover of DE"""
        child = parent.copy()
        dim = parent.size
        n = rng.randint(0, dim)
        L = 0
        while (rng.rand() < crossover_prob) and L < dim:
            idx = (n + L) % dim
            child[idx] = mutant[idx]
            L = L + 1

        return child

    @classmethod
    def crossover_func(
        cls,
        crossover_strategy: str,
        parent: np.ndarray,
        mutant: np.ndarray,
        crossover_prob: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Performs DE crossover"""
        if crossover_strategy not in cls.strategy_choices:
            raise ValueError(f"crossover_strategy must be in {cls.strategy_choices}, but got {crossover_strategy}")

        params = dict(crossover_prob=crossover_prob, parent=parent, mutant=mutant, rng=rng)
        child = getattr(cls, f"_crossover_{crossover_strategy}")(**params)
        return child
