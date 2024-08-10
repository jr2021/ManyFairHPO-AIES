from typing import Union, Dict, List, Any, Tuple

import time

import ConfigSpace as CS
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from utils import (
    get_config_space,
    get_dataset,
    get_encoder,
    get_protected_attribute,
    get_strat,
    get_model,
    get_split,
    get_sample
)
import pickle as pkl
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

MODEL_KEY = None

class FairMOHPOBenchmark(AbstractMultiObjectiveBenchmark):
    def __init__(self,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        global MODEL_KEY
        MODEL_KEY = kwargs['model_name']

        super(FairMOHPOBenchmark, self).__init__(rng=rng, **kwargs)
        self.socket_id = rng

    def init(self, settings, seed):
        self.obj_keys = settings['objectives']
        self.all_keys = ['f1', 'ddsp', 'deop', 'deod', 'invd', 'comp']
        self.meta_keys = set(self.all_keys)-set(self.obj_keys)
        self.seed = seed
        
        with open('/work/dlclarge2/robertsj-fairmohpo/bounds.pkl', 'rb') as f:
            self.bounds = pkl.load(f)[settings['dataset_name']]

        self.dataset_key = settings['dataset_name']
        self.model_key = settings['model_name']
        self.prot_attr = get_protected_attribute(self.dataset_key)


        self.X, self.y = get_dataset(self.dataset_key)
        self.encoder = get_encoder(self.X)
        self.X = pd.DataFrame(
            self.encoder.fit_transform(self.X), index=self.X.index
        )

        self.inner_kfold = StratifiedKFold(n_splits=3)
        self.outer_kfold = StratifiedKFold(n_splits=5)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:

        configuration_space = get_config_space(model_key=MODEL_KEY, seed=seed)
        return configuration_space

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'budget', lower=1, upper=2, default_value=1, log=False
            )
        )
        return fidelity_space

    @staticmethod
    def get_objective_names() -> List[str]:
        pass

    @staticmethod
    def get_meta_information() -> Dict:
        pass

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, 
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        logger.info(f"evaluating configuration: {configuration}")
        start = time.time()

        model = get_model(
            config=configuration, 
            model_key=self.model_key, 
            n_features=self.X.shape[1]
        )

        prot_column = self.y.index.to_frame()[self.prot_attr].values
        outer_strat = get_strat(self.y.values, prot_column)

        val_samples, test_samples, meta_samples, obj_samples = [], [], [], []
        for train_val_idx, test_idx in self.outer_kfold.split(self.X, outer_strat):
            outer_split = get_split(self.X, self.y, train_val_idx, test_idx)        
            X_train_val, y_train_val, X_test, y_test, = outer_split

            prot_column = y_train_val.index.to_frame()[self.prot_attr].values
            inner_strat = get_strat(y_train_val.values, prot_column)
            
            for train_idx, val_idx in self.inner_kfold.split(X_train_val, inner_strat):
                inner_split = get_split(X_train_val, y_train_val, train_idx, val_idx)
                X_train, y_train, X_val, y_val = inner_split

                model.fit(X_train, y_train.values)

                obj_sample = get_sample(model, self.obj_keys, X_val, y_val, self.prot_attr, self.bounds)
                obj_samples.append(obj_sample)

                meta_sample = get_sample(model, self.meta_keys, X_val, y_val, self.prot_attr, self.bounds)
                meta_samples.append(meta_sample)

                val_sample = get_sample(model, self.all_keys, X_val, y_val, self.prot_attr, self.bounds)
                val_samples.append(val_sample)

                test_sample = get_sample(model, self.all_keys, X_test, y_test, self.prot_attr, self.bounds)
                test_samples.append(test_sample)

        val_means = np.mean(val_samples, axis=0)
        obj_means = np.mean(obj_samples, axis=0)
        meta_means = np.mean(meta_samples, axis=0)
        test_means = np.mean(test_samples, axis=0)
        

        if obj_means[0] == 0:
            logger.warning("DEGENERATE CONFIGURATION")
            obj_means = np.ones(len(self.obj_keys))
            obj_means[0] = 0
            meta_means = np.ones(len(self.meta_keys))
            test_means = np.ones(len(self.all_keys))
            test_means[0] = 0

        cost = time.time() - start

        result = {'function_value': {f'val_{obj_key}': obj_means[i] for i, obj_key in enumerate(self.obj_keys)},
                    'cost': cost,
                    'info': {}
        }

        for i, obj_key in enumerate(self.obj_keys):
            result['info'][f"val_{obj_key}"] = obj_means[i]

        for i, meta_key in enumerate(self.meta_keys):
            result['info'][f"val_{meta_key}"] = meta_means[i]

        for i, key in enumerate(self.all_keys):
            result['info'][f"test_{key}"] = test_means[i]

        logger.info(f"function value: {result}")

        return result  

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, 
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        logger.info(f"evaluating configuration: {configuration}")
        start = time.time()

        model = get_model(
            config=configuration, 
            model_key=self.model_key, 
            n_features=self.X.shape[1]
        )

        prot_column = self.y.index.to_frame()[self.prot_attr].values
        outer_strat = get_strat(self.y.values, prot_column)

        val_samples, test_samples, meta_samples = [], [], []
        for train_val_idx, test_idx in self.outer_kfold.split(self.X, outer_strat):
            outer_split = get_split(self.X, self.y, train_val_idx, test_idx)        
            X_train_val, y_train_val, X_test, y_test, = outer_split

            prot_column = y_train_val.index.to_frame()[self.prot_attr].values
            inner_strat = get_strat(y_train_val.values, prot_column)
            
            for train_idx, val_idx in self.inner_kfold.split(X_train_val, inner_strat):
                inner_split = get_split(X_train_val, y_train_val, train_idx, val_idx)
                X_train, y_train, X_val, y_val = inner_split

                model.fit(X_train, y_train.values)

                val_sample = get_sample(model, self.obj_keys, X_val, y_val, self.prot_attr)
                val_samples.append(val_sample)

                meta_sample = get_sample(model, self.meta_keys, X_val, y_val, self.prot_attr)
                meta_samples.append(meta_sample)

                test_sample = get_sample(model, self.all_keys, X_test, y_test, self.prot_attr)
                test_samples.append(test_sample)

        val_means = np.mean(val_samples, axis=0)
        meta_means = np.mean(meta_samples, axis=0)
        test_means = np.mean(test_samples, axis=0)

        if np.all(val_means==0):
            logger.warning("DEGENERATE CONFIGURATION")
            val_means = np.ones(len(self.obj_keys))
            val_means[0] = 0
            meta_means = np.ones(len(self.meta_keys))
            test_means = np.ones(len(self.all_keys))
            test_means[0] = 0

        cost = time.time() - start

        result = {'function_value': {f'val_{obj_key}': val_means[i] for i, obj_key in enumerate(self.obj_keys)},
                    'cost': cost,
                    'info': {}
        }

        for i, obj_key in enumerate(self.obj_keys):
            result['info'][f"val_{obj_key}"] = val_means[i]

        for i, meta_key in enumerate(self.meta_keys):
            result['info'][f"val_{meta_key}"] = meta_means[i]

        for i, key in enumerate(self.all_keys):
            result['info'][f"test_{key}"] = test_means[i]

        logger.info(f"function value: {result}")

        return result  

__all__ = ['FairMOHPOBenchmark']