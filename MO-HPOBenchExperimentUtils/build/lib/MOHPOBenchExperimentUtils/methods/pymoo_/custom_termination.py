from abc import abstractmethod

from pymoo.util.sliding_window import SlidingWindow
from pymoo.util.termination.collection import TerminationCollection
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.util.termination.f_tol import calc_delta_norm, normalize, IGD


class CustomSlidingWindowTermination(TerminationCollection):
    """
    Patch to add time constraint to termination class -
    Following: https://github.com/anyoptimization/pymoo/pull/284/files
    """
    def __init__(self,
                 metric_window_size=None,
                 data_window_size=None,
                 min_data_for_metric=1,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 max_time=None,
                 truncate_metrics=True,
                 truncate_data=True,
                 ) -> None:
        """

        Parameters
        ----------

        metric_window_size : int
            The last generations that should be considering during the calculations

        data_window_size : int
            How much of the history should be kept in memory based on a sliding window.

        nth_gen : int
            Each n-th generation the termination should be checked for

        """

        super().__init__(MaximumGenerationTermination(n_max_gen=n_max_gen),
                         MaximumFunctionCallTermination(n_max_evals=n_max_evals),
                         TimeBasedTermination(max_time=max_time))

        # the window sizes stored in objects
        self.data_window_size = data_window_size
        self.metric_window_size = metric_window_size

        # the obtained data at each iteration
        self.data = SlidingWindow(data_window_size) if truncate_data else []

        # the metrics calculated also in a sliding window
        self.metrics = SlidingWindow(metric_window_size) if truncate_metrics else []

        # each n-th generation the termination decides whether to terminate or not
        self.nth_gen = nth_gen

        # number of entries of data need to be stored to calculate the metric at all
        self.min_data_for_metric = min_data_for_metric

    def _do_continue(self, algorithm):

        # if the maximum generation or maximum evaluations say terminated -> do so
        if not super()._do_continue(algorithm):
            return False

        # store the data decided to be used by the implementation
        obj = self._store(algorithm)
        if obj is not None:
            self.data.append(obj)

        # if enough data has be stored to calculate the metric
        if len(self.data) >= self.min_data_for_metric:
            metric = self._metric(self.data[-self.data_window_size:])
            if metric is not None:
                self.metrics.append(metric)

        # if its the n-th generation and enough metrics have been calculated make the decision
        if algorithm.n_gen % self.nth_gen == 0 and len(self.metrics) >= self.metric_window_size:

            # ask the implementation whether to terminate or not
            return self._decide(self.metrics[-self.metric_window_size:])

        # otherwise by default just continue
        else:
            return True

    # given an algorithm object decide what should be stored as historical information - by default just opt
    def _store(self, algorithm):
        return algorithm.opt

    @abstractmethod
    def _decide(self, metrics):
        pass

    @abstractmethod
    def _metric(self, data):
        pass

    def get_metric(self):
        if len(self.metrics) > 0:
            return self.metrics[-1]
        else:
            return None


class CustomMultiObjectiveSpaceToleranceTermination(CustomSlidingWindowTermination):

    def __init__(self,
                 tol=0.0025,
                 n_last=30,
                 nth_gen=5,
                 n_max_gen=None,
                 n_max_evals=None,
                 max_time=None,
                 **kwargs) -> None:
        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         max_time=max_time,
                         **kwargs)
        self.tol = tol

    def _store(self, algorithm):
        F = algorithm.opt.get("F")
        return {
            "ideal": F.min(axis=0),
            "nadir": F.max(axis=0),
            "F": F
        }

    def _metric(self, data):
        last, current = data[-2], data[-1]

        # this is the range between the nadir and the ideal point
        norm = current["nadir"] - current["ideal"]

        # if the range is degenerated (very close to zero) - disable normalization by dividing by one
        norm[norm < 1e-32] = 1

        # calculate the change from last to current in ideal and nadir point
        delta_ideal = calc_delta_norm(current["ideal"], last["ideal"], norm)
        delta_nadir = calc_delta_norm(current["nadir"], last["nadir"], norm)

        # get necessary data from the current population
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]

        # normalize last and current with respect to most recent ideal and nadir
        c_N = normalize(c_F, c_ideal, c_nadir)
        l_N = normalize(last["F"], c_ideal, c_nadir)

        # calculate IGD from one to another
        delta_f = IGD(c_N).do(l_N)

        return {
            "delta_ideal": delta_ideal,
            "delta_nadir": delta_nadir,
            "delta_f": delta_f
        }

    def _decide(self, metrics):
        delta_ideal = [e["delta_ideal"] for e in metrics]
        delta_nadir = [e["delta_nadir"] for e in metrics]
        delta_f = [e["delta_f"] for e in metrics]
        return max(max(delta_ideal), max(delta_nadir), max(delta_f)) > self.tol
