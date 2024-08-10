from typing import List
import numpy as np
from pygmo import hypervolume
from loguru import logger
import sys


def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto


def pareto_index(costs: np.ndarray, index_list):

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):

        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self

    index_return = index_list[is_pareto]

    return is_pareto, index_return


def nDS_index(costs, index_list):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :list of indeces
    :return: list of all fronts, sorted indeces
    """

    dominating_list = []
    index_return_list = []
    fronts = []
    while costs.size > 0:
        dominating, index_return = pareto_index(costs, index_list)
        fronts.append(costs[dominating])
        costs = costs[~dominating]
        index_list = index_list[~dominating]
        dominating_list.append(dominating)
        index_return_list.append(index_return)

    return fronts, index_return_list


def crowdingDist(fronts, index_list):
    """
    Implementation of the crowding distance
    :param front: (n_points, m_cost_values) array
    :return: sorted_front and corresponding distance value of each element in the sorted_front
    """
    dist_list = []
    index_return_list = []

    for g in range(len(fronts)):
        front = fronts[g]
        index_ = index_list[g]
        distances = np.zeros((front.shape[0]))  # Distance for each configuration.

        # Iterate over all objectivs
        for i_obj in range(front.shape[1]):
            # Sort by the objective value.
            ordering = np.argsort(front[:, i_obj], axis=0)
            sorted_front = front[ordering]
            sorted_distances = distances[ordering]

            # We add a normalization factor to scale each objective
            norm_factor = sorted_front[-1, i_obj] - sorted_front[0, i_obj]
            # calculate each individual's Crowding Distance of i-th objective
            # technique: shift the list and zip
            # distances[ordering[[0, -1]]] = np.inf
            sorted_distances[[0, -1]] = np.inf

            # ordering[1:-1]
            for cur_index, (prev, cur, next) in enumerate(zip(sorted_front[:-2], sorted_front[1:-1], sorted_front[2:])):
                sorted_distances[cur_index + 1] += (next[i_obj] - prev[i_obj]) / norm_factor # sum up the distance of ith individual along each of the objectives

            # Add the calculated distances to the overall distances.
            # Reorder distances since the sorted_distance vector is sorted wrt the observations (and their objective values)
            remap_order = np.zeros_like(ordering)
            for i, val in enumerate(ordering):
                remap_order[val] = i
            resorted_distances = sorted_distances[remap_order]
            distances += resorted_distances

        logger.debug(f'Front {g+1} | {len(fronts)}: Distances: {distances}')

        ordering_by_distances = np.argsort(distances)
        ordering_by_distances = ordering_by_distances[::-1]  # reverse

        sorted_front_by_dist = front[ordering_by_distances]
        sorted_index_max_by_dist = index_[ordering_by_distances]
        sorted_distances_by_dist = distances[ordering_by_distances]

        dist_list.append((sorted_front_by_dist, sorted_distances_by_dist))
        index_return_list.append(sorted_index_max_by_dist)

    return dist_list, index_return_list


def nDS(costs: np.ndarray):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :return: list of all fronts
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # Stepwise compute the pareto front without all prior dominating points
    my_costs = costs.copy()
    remain = np.ones(len(my_costs), dtype=np.bool)
    fronts = []
    while np.any(remain):
        front_i = pareto(my_costs)
        fronts.append(my_costs[front_i, :])
        my_costs[front_i, :] = np.inf
        remain = np.logical_and(remain, np.logical_not(front_i))
    return fronts


def get_ref_point(costs):
    max_values = np.max(costs, axis=0)
    ref_point = np.maximum(
            1.1 * max_values,  # case: value > 0
            0.9 * max_values  # case: value < 0
        )
    #to handle 0 values
    if np.all((np.array(ref_point) == 0.0)):
        ref_point = np.full(len(max_values), sys.float_info.epsilon)
    logger.debug(f'curr reference point:{ref_point}')
    return ref_point


def contributionsHV3D(costs, ref_point=None):
    if ref_point is None:
        ref_point = get_ref_point(costs)
    hv = hypervolume(costs)
    return hv.contributions(ref_point)


def computeHV(costs, ref_point=None):
    if ref_point is None:
        ref_point = get_ref_point(costs)
    hv = hypervolume(costs)
    return hv.compute(ref_point)


def computeHV2D(front: np.ndarray, ref: List[float]):
    """
    Compute the Hypervolume for the pareto front  (only implement it for 2D)
    :param front: (n_points, m_cost_values) array for which to compute the volume
    :param ref: coordinates of the reference point
    :returns: Hypervolume of the polygon spanned by all points in the front + the reference point
    """

    front = np.asarray(front)
    assert front.ndim == 2
    assert len(ref) == 2

    # We assume all points already sorted
    list_ = [ref]
    for x in front:
        elem_at = len(list_) - 1
        list_.append([list_[elem_at][0], x[1]])  # add intersection points by keeping the x constant
        list_.append(x)
    list_.append([list_[-1][0], list_[0][1]])
    sorted_front = np.array(list_)

    def shoelace(x_y):  # taken from https://stackoverflow.com/a/58515054
        x_y = np.array(x_y)
        x_y = x_y.reshape(-1, 2)

        x = x_y[:, 0]
        y = x_y[:, 1]

        S1 = np.sum(x * np.roll(y, -1))
        S2 = np.sum(y * np.roll(x, -1))

        area = .5 * np.absolute(S1 - S2)

        return area

    return shoelace(sorted_front)
