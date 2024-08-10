import enum
import sys
import uuid
from copy import deepcopy
from typing import Optional, Dict
from MOHPOBenchExperimentUtils.utils import adapt_configspace_configuration_to_ax_space
import numpy as np
from ax import Experiment, GeneratorRun, Arm
from loguru import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


class Recombination(enum.IntEnum):
    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    INTERMEDIATE = 1  # intermediate recombination


class Mutation(enum.IntEnum):
    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation


class ParentSelection(enum.IntEnum):
    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


class Member:
    """
    Class to simplify member handling.
    """

    def __init__(self, search_space,
                 budget: int,
                 mutation: Mutation,
                 recombination: Recombination,
                 name_file: Optional[str] = 'eash',
                 sigma: Optional[float] = None,
                 recom_prob: Optional[float] = None,
                 x_coordinate: Optional[Dict] = None,
                 experiment: Experiment = None,
                 adapt_cfg_function=None) -> None:
        """
        Init
        :param initial_x: Initial coordinate of the member
        :param target_function: The target function that determines the fitness value
        :param mutation: hyperparameter that determines which mutation type use
        :param recombination: hyperparameter that determines which recombination type to use
        :param sigma: Optional hyperparameter that is only active if mutation is gaussian
        :param recom_prob: Optional hyperparameter that is only active if recombination is uniform
        """
        self._space = search_space
        self._id = uuid.uuid4()
        self._name_file = name_file
        self._x = search_space.config_space.sample_configuration().get_dictionary() if not x_coordinate else x_coordinate
        self._age = 0  # basically indicates how many offspring were generated from this member
        self._mutation = mutation
        self._recombination = recombination
        self._x_changed = True
        self._fit = None
        self._sigma = sigma
        self._recom_prob = recom_prob
        self._budget = budget
        self._experiment = experiment
        self._num_evals = 0
        self.adapt_cfg = adapt_cfg_function

    @property  # fitness can only be queried never set
    def fitness(self):
        if self._x_changed:  # Only if the x_coordinate or the budget has changed we need to evaluate the fitness.
            self._x_changed = False
            print(type(self._x_changed))
            logger.debug(f"config:{self._x}")

            params = deepcopy(self._x)

            print(type(params))
            params['budget'] = int(self._budget)
            trial_name = '{}-{}'.format(self._id, self._num_evals)
            params['id'] = trial_name
            params = adapt_configspace_configuration_to_ax_space(params, search_space=self._space)

            logger.debug(f"params:{params}")
            trial = self._experiment.new_trial(GeneratorRun([Arm(params, name=params['id'])]))
            data = self._experiment.eval_trial(trial)

            objectives = self._experiment.optimization_config.objective.metrics
            logger.debug(f"objectives to fetch results for :{objectives}")
            metrics = [obj.name for obj in objectives]
            logger.debug(f"metrices:{metrics}")
            metrics = [float(data.df[data.df['metric_name'] == obj]['mean']) for obj in metrics]
            logger.debug(f"metrices values:{metrics}")

            self._fit = metrics
            self._num_evals += 1

        return self._fit  # otherwise we can return the cached value

    @property  # properties let us easily handle getting and setting without exposing our private variables
    def x_coordinate(self):
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value):
        self._x_changed = True
        self._x = value

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        self._x_changed = True
        self._budget = value

    @property
    def id(self):
        return self._id

    def mutate(self):
        """
        Mutation which creates a new offspring
        :return: new member who is based on this member
        """
        new_x = self.x_coordinate.copy()

        # self.logger.debug('new point before mutation:')
        # self.logger.debug(new_x)

        if self._mutation == Mutation.UNIFORM:
            keys = np.random.choice(list(new_x.keys()), 5, replace=False)
            for k in keys:
                k = str(k)
                if self._space.is_mutable_hyperparameter(k):
                    new_x[k] = self._space.sample_hyperparameter(k)

        elif self._mutation != Mutation.NONE:
            # We won't consider any other mutation types
            raise NotImplementedError

        child = Member(self._space, self._budget, self._mutation, self._recombination, self._name_file,
                       self._sigma, self._recom_prob, new_x, self._experiment)
        self._age += 1
        return child

    def recombine(self, partner):
        """
        Recombination of this member with a partner
        :param partner: Member
        :return: new offspring based on this member and partner
        """
        # if self._recombination == Recombination.INTERMEDIATE:
        #    new_x = 0.5 * (self.x_coordinate + partner.x_coordinate)
        if self._recombination == Recombination.UNIFORM:
            assert self._recom_prob is not None, \
                'for this recombination type you have to specify the recombination probability'

            new_x = self.x_coordinate.copy()
            for k in new_x.keys():
                if (np.random.rand() >= self._recom_prob) and (k in partner.x_coordinate.keys()) \
                        and (self._space.is_mutable_hyperparameter(k)):
                    new_x[k] = partner.x_coordinate[k]

        elif self._recombination == Recombination.NONE:
            new_x = self.x_coordinate.copy()  # copy is important here to not only get a reference
        else:
            raise NotImplementedError

        child = Member(self._space, self._budget, self._mutation, self._recombination, self._name_file,
                       self._sigma, self._recom_prob, new_x, self._experiment)
        self._age += 1
        return child

    def __str__(self):
        """Makes the class easily printable"""
        str = "Population member: Age={}, budget={}, x={}, f(x)={}".format(self._age, self._budget, self.x_coordinate,
                                                                           self.fitness)
        return str

    def __repr__(self):
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + '\n'

