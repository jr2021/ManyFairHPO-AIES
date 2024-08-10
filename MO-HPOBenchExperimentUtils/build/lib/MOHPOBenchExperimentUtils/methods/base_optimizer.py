from typing import Dict, Union

import ConfigSpace as CS


class Optimizer:
    def __init__(
            self,
            optimizer_settings: Dict,
            benchmark_settings: Dict,
            configspace: CS.ConfigurationSpace,
            **kwargs: Union[None, Dict],
    ):

        self.optimizer_settings = optimizer_settings
        self.benchmark_settings = benchmark_settings
        self.cs = configspace

    def init(self, **kwargs):
        raise NotImplementedError()

    def setup(self, **kwargs):
        raise NotImplementedError()

    def run(self, **kwargs):
        raise NotImplementedError()
