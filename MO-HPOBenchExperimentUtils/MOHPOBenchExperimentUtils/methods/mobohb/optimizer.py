import ConfigSpace as CS
from MOHPOBenchExperimentUtils.methods.base_optimizer import Optimizer

from typing import Dict, Union
from MOHPOBenchExperimentUtils.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
from loguru import logger

from baselines.methods.mobohb.hpbandster.core.nameserver import NameServer
from baselines.methods.mobohb.hpbandster.optimizers.mobohb import MOBOHB
from baselines.methods.mobohb.mobohb_worker import MOBOHBWorker


class MOBOHBOptimizer(Optimizer):

    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configspace: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):
        super(MOBOHBOptimizer, self).__init__(optimizer_settings, benchmark_settings, configspace, **kwargs)

        self.algorithm = None
        self.experiment = None
        self._objective_function = None
        self.scenario = None
        self.seed = 0
        self.fidelities: Union[Dict, None] = None

    def init(self, experiment: MultiObjectiveSimpleExperiment, **kwargs):

        self.experiment = experiment
        self.fidelities: Dict[str, Union[int, float]] = \
            {entry['name']: entry['limits'] for entry in self.benchmark_settings['fidelity']}

    def setup(self, seed: int = 0, **kwargs):
        self.seed = seed

    def run(self, **kwargs):

        logger.info('Start Optimization Run')
        ns = NameServer(run_id=str(self.seed), host='127.0.0.1', port=0)
        ns_host, ns_port = ns.start()

        # We only use a single local worker.
        w = MOBOHBWorker(
            experiment=self.experiment,
            search_space=self.experiment.cs_search_space,
            seed=self.seed,
            run_id=str(self.seed),
            host='127.0.0.1',
            nameserver=ns_host,
            nameserver_port=ns_port
        )

        logger.info(f'Finished Optimization Run with incumbent {incumbent}')
