from typing import Dict, Union
import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
from fairmohpo.benchmark import FairMOHPOBenchmark

Hyperparameter_Type = Union[float, int]

class FairMOHPO:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.generations = config.generations
        self.pop_size = config.pop_size
        self.seed = config.seed
        self.config_file = config.config_file
        self.log_dir = config.log_dir

        

        self.optimizer = Pymoo_NSGA3()

    def run(self):
        # TODO: run NSGA-II
        pass   

    def save(self):
        try:
            os.makedirs(self.log_dir)
            print(f"Created logging directory: {self.log_dir}")
        except FileExistsError:
            print(f"Overwrote existing logging directory: {self.log_dir}")

        with open(f"{self.log_dir}/fairmohpo.pkl", 'wb') as f:
            pickle.dump(self, f)