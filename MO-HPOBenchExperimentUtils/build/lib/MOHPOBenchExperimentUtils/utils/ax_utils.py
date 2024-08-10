import ax
from typing import Dict, Tuple, List, Union
from MOHPOBenchExperimentUtils.utils.hpobench_utils import HPOBenchMetrics


def get_ax_metrics_from_metric_dict(metric_dict: Dict) \
        -> Tuple[ax.MultiObjective, List[ax.ObjectiveThreshold], Union[List[ax.Metric], None]]:
    targets = []
    thresholds = []

    for entry in metric_dict['target']:
        target = ax.Metric(entry['name'], lower_is_better=entry['lower_is_better'])
        threshold = ax.ObjectiveThreshold(target, entry['threshold'])
        targets.append(target)
        thresholds.append(threshold)

    multi_objective = ax.MultiObjective(targets)

    if 'additional' in metric_dict:
        extra_metrics = [ax.Metric(name=entry['name'],
                                   lower_is_better=entry.get('lower_is_better', None))
                         for entry in metric_dict['additional']]

        # HPOBench always returns a cost field.
        if HPOBenchMetrics.COST not in metric_dict.keys():
            extra_metrics.append(ax.Metric(name=HPOBenchMetrics.COST.value, lower_is_better=True))

        if HPOBenchMetrics.WALLCLOCK_CONFIG_START not in metric_dict.keys():
            extra_metrics.append(ax.Metric(name=HPOBenchMetrics.WALLCLOCK_CONFIG_START.value))

        if HPOBenchMetrics.WALLCLOCK_CONFIG_END not in metric_dict.keys():
            extra_metrics.append(ax.Metric(name=HPOBenchMetrics.WALLCLOCK_CONFIG_END.value))

    else:
        extra_metrics = None

    return multi_objective, thresholds, extra_metrics
