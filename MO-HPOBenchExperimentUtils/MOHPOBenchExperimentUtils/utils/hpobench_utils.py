from importlib import import_module
from typing import Any
from loguru import logger as _log
from enum import Enum


class HPOBenchMetrics(str, Enum):

    COST = 'cost'
    WALLCLOCK_CONFIG_START = 'EXP_WALLCLOCK_CONFIG_START'
    WALLCLOCK_CONFIG_END = 'EXP_WALLCLOCK_CONFIG_END'

    def __str__(self):
        return str(self.value)


def load_benchmark(benchmark_name, import_from, use_local: bool) -> Any:
    """
    Load the benchmark object.
    If not `use_local`:  Then load a container from a given source, defined in the HPOBench.
    Import via command from hpobench.[container.]benchmarks.<import_from> import <benchmark_name>
    Parameters
    ----------
    benchmark_name : str
    import_from : str
    use_local : bool
        By default this value is set to false.
        In this case, a container will be downloaded. This container includes all necessary files for the experiment.
        You don't have to install something.
        If true, use the experiment locally. Therefore the experiment has to be installed.
        See the experiment description in the HPOBench.
    Returns
    -------
    Benchmark
    """
    import_str = 'hpobench.' + ('container.' if not use_local else '') + 'benchmarks.' + import_from
    _log.debug(f'Try to execute command: from {import_str} import {benchmark_name}')

    module = import_module(import_str)
    benchmark_obj = getattr(module, benchmark_name)
    _log.debug(f'Benchmark {benchmark_name} successfully loaded')

    return benchmark_obj
