import warnings
warnings.filterwarnings("ignore")
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from sklearn.metrics import f1_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from loguru import logger
from fast_pareto import is_pareto_front
from pygmo import hypervolume
import pandas as pd
from numpy.random import RandomState
import xgboost as xgb
from torch import nn
import torch
from aif360.sklearn.datasets import (
    fetch_adult,
    fetch_compas, 
    fetch_german,
    fetch_bank,
    fetch_lawschool_gpa
)
from aif360.sklearn.metrics import (
    equal_opportunity_difference, 
    generalized_entropy_error,
    average_odds_difference,
    statistical_parity_difference,
    between_group_generalized_entropy_error
)

import pandas as pd
import ConfigSpace as CS
import numpy as np
import yaml
from yaml.loader import SafeLoader
from MOHPOBenchExperimentUtils.utils.experiment_utils import load_experiment
from ax.core.base_trial import TrialStatus

pd.options.mode.chained_assignment = None

def get_model(config, model_key, n_features):
    if model_key == 'rf':
        max_features = int(np.rint(np.power(n_features, config["max_features"])))
        model = RandomForestClassifier(
            max_depth=int(config['max_depth']),
            min_samples_split=int(config['min_samples_split']), 
            min_samples_leaf=int(config['min_samples_leaf']),
            max_features=max_features,
            n_estimators=int(config['n_estimators']),
            bootstrap=True,
        )
        return model
    elif model_key == 'nn':
        depth, width = int(config['depth']), int(config['width'])
        hidden_layers = [width] * depth
        model = MLPClassifier(
            batch_size=int(config['batch_size']),
            alpha=config['alpha'],
            learning_rate_init=config['learning_rate_init'],
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            early_stopping=True,
            n_iter_no_change=int(config['n_iter_no_change']),
        )
        return model
    elif model_key == 'xgb':
        model = xgb.XGBClassifier(
            eta=config['eta'],
            max_depth=int(config['max_depth']),
            colsample_bytree=config['colsample_bytree'],
            reg_lambda=config['reg_lambda'],
            n_estimators=int(config['n_estimators']),
            booster='gbtree',
            objective='binary:logistic'
        )
        return model
    elif model_key == 'hgb':
        model = HistGradientBoostingClassifier(
            max_depth=int(config['max_depth']),
            max_leaf_nodes=int(config['max_leaf_nodes']),
            learning_rate=config['learning_rate'],
            l2_regularization=config['l2_regularization'],
            max_iter=config['n_estimators'],
            early_stopping=False,
        )
        return model
    elif model_key == 'svm':
        model = SVC(
            C=config['C'],
            gamma=config['gamma']
        )
        return model

def get_default_model(model_key):
    if model_key == 'rf':
        model = RandomForestClassifier()
        return model
    elif model_key == 'nn':
        model = MLPClassifier()
        return model
    elif model_key == 'xgb':
        model = xgb.XGBClassifier()
        return model

class MyModule(nn.Module):
    def __init__(self, depth, width, in_features):
        super().__init__()

        hidden = nn.ModuleList()
        for _ in range(depth):
            hidden.append(nn.Linear(width, width))
            hidden.append(nn.ReLU())

        self.model = nn.Sequential(
            nn.Linear(in_features, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2),
            nn.Sigmoid()
        )

    def forward(self, x, **kwargs):
        return self.model(x.float())

def get_dataset(dataset_key):

    rng = RandomState(0)

    if dataset_key == 'adult':
        X, y, _ = fetch_adult(
            binary_race=True,
            dropna=True,
            dropcols=[get_protected_attribute('adult'), 'education', 'sex', 'native-country', 'relationship']
        )

        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm]

        # encode target variable
        target_map = {'<=50K': 0, '>50K': 1}
        y = y.map(target_map)
        
        # encode protected attributes
        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
        
        return X, y

    elif dataset_key == 'compas':
        X, y = fetch_compas(
            binary_race=True,
            dropna=True,
            numeric_only=False,
            dropcols=[get_protected_attribute('compas'), 'age_cat', 'sex', 'c_charge_desc']
        )
    
        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm] 

        # encode protected attributes
        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

        # encode target variable and flip b/c recidivism is an unfavourable outcome
        target_map = {'Survived': 1, 'Recidivated': 0}
        y = y.map(target_map)
        
        return X, y

    elif dataset_key == 'german':
        X, y = fetch_german(
            binary_age=True, 
            dropna=True, 
            numeric_only=False,
            dropcols=[get_protected_attribute('german')],
        )

        # encode categoricals
        emp_map =  {
            'unemployed': 0,
            '1<=X<4': 2, 
            '<1': 1,
            '>=7': 4,
            '4<=X<7': 3 
        }

        sav_map =  {
            '<100': 1,
            '100<=X<500': 2,
            '>=1000': 4,
            '500<=X<1000': 3, 
            'no known savings': 0
        }

        X['employment'] = X['employment'].map(emp_map).astype('int')
        X['savings_status'] = X['savings_status'].map(sav_map).astype('int')

        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm]

        # encode target variable
        target_map = {'good': 1, 'bad': 0}
        y = y.map(target_map)
        
        # encode protected attributes
        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
        
        return X, y
    elif dataset_key == 'bank':

        numeric_only = False
        X, y = fetch_bank(
            dropna=True, 
            numeric_only=numeric_only,
            dropcols=[get_protected_attribute('bank')]
        )

        month_map =  {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9, 
            'oct': 10,
            'nov': 11,
            'dec': 12
        }
        X['month'] = X['month'].map(month_map).astype('int')

        edu_map =  {
            'primary': 0,
            'secondary': 1,
            'tertiary': 2,
        }
        X['education'] = X['education'].map(edu_map).astype('int')

        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm]

        # encode target variable
        if not numeric_only:
            target_map = {'yes': 1, 'no': 0}
            y = y.map(target_map)
        
        # encode protected attributes
        prot_attr = get_protected_attribute('bank')
        X.index = pd.CategoricalIndex(data=[0 if val < 35 else 1 for val in X.index.values], name=prot_attr)
        y.index = pd.CategoricalIndex(data=[0 if val < 35 else 1 for val in y.index.values], name=prot_attr)  
        
        return X, y

    elif dataset_key == 'lawschool':
        dataset = fetch_lawschool_gpa(
            dropna=True,
            dropcols=[get_protected_attribute('lawschool')]
        )

        X, y = dataset.X, dataset.y

        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm]

        # discretize target variable
        top_quartile = 0.740000
        y.iloc[:] = y.values > np.mean(y.values)

        # encode protected attributes
        prot_map = {'white': 1, 'black': 0}
        X.index = X.index.map(prot_map)
        y.index = y.index.map(prot_map)

        X.index = pd.MultiIndex.from_arrays([X.index.codes], names=X.index.names)
        y.index = pd.MultiIndex.from_arrays([y.index.codes], names=y.index.names)
        
        return X, y

    else:
        raise KeyError('dataset is not supported')

def get_config_space(model_key, seed):
    if model_key == 'rf':
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=50, default_value=10, log=True),
            CS.UniformIntegerHyperparameter('min_samples_split', lower=2, upper=128, default_value=32, log=True),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=20, default_value=1),
            CS.UniformFloatHyperparameter('max_features', lower=0, upper=1, default_value=0.5),
            CS.UniformIntegerHyperparameter('n_estimators', lower=1, upper=200, default_value=100, log=True)
        ])

        return cs
        
    elif model_key == 'nn':
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('depth', lower=1, upper=3, default_value=3),
            CS.UniformIntegerHyperparameter('width', lower=16, upper=1024, default_value=64, log=True),
            CS.UniformIntegerHyperparameter('batch_size', lower=4, upper=256, default_value=32, log=True),
            CS.UniformFloatHyperparameter('alpha', lower=10**-8, upper=1, default_value=10**-3, log=True),
            CS.UniformFloatHyperparameter('learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True),
            CS.UniformIntegerHyperparameter('n_iter_no_change', lower=1, upper=20, default_value=10, log=True)
        ])

        return cs

    elif model_key == 'svm':
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter("C", 2**-10, 2**10, log=True, default_value=1.0),
            CS.UniformFloatHyperparameter("gamma", 2**-10, 2**10, log=True, default_value=0.1)
        ])
        return cs

    elif model_key == 'xgb':
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter('eta', lower=2**-10, upper=1.0, default_value=0.3, log=True),
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=50, default_value=10, log=True),
            CS.UniformFloatHyperparameter('colsample_bytree', lower=0.1, upper=1.0, default_value=1.0),
            CS.UniformFloatHyperparameter('reg_lambda', lower=2**-10, upper=2**10, default_value=1.0, log=True),
            CS.UniformIntegerHyperparameter('n_estimators', lower=1, upper=200, default_value=100, log=True)
        ])

        return cs
    
    elif model_key == 'hgb':
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('max_depth', lower=6, upper=30, default_value=6, log=True),
            CS.UniformIntegerHyperparameter('max_leaf_nodes', lower=2, upper=64, default_value=32, log=True),
            CS.UniformFloatHyperparameter('learning_rate', lower=2**-10, upper=1, default_value=0.1, log=True),
            CS.UniformFloatHyperparameter( 'l2_regularization', lower=2**-10, upper=2**10, default_value=0.1, log=True),
            CS.UniformIntegerHyperparameter('n_estimators', lower=1, upper=200, default_value=100, log=True)
        ])

        return cs

    else:
        raise KeyError('model is not supported')

def get_encoder(X_train):
    pipeline = make_column_transformer(
        (
            OneHotEncoder(
                sparse=False,
                drop='if_binary',
                handle_unknown='error'
            ), 
            X_train.dtypes == 'category'
        ),
        remainder=MinMaxScaler()
    )
    return pipeline

def get_split(X, y, train_idx, val_idx):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    return X_train, y_train, X_val, y_val

def get_sample(model, obj_keys, X, y, prot_attr, bounds):
    y_pred = model.predict(X.values)
    
    sample = np.zeros(len(obj_keys))
    for i, obj_key in enumerate(obj_keys):
         sample[i] = get_score(obj_key, y_pred, y, prot_attr, bounds)
    
    return sample

def get_egr_sample(model, obj_keys, dataset, y, prot_attr):
    y_pred = model.predict(dataset).labels
    
    sample = np.zeros(len(obj_keys))
    for i, obj_key in enumerate(obj_keys):
         sample[i] = get_score(obj_key, y_pred, y, prot_attr)
    
    return sample

def inverse_distance(y_true, y_pred, prot_attr):
    prot_col = y_true.index.to_frame()[prot_attr].astype(int).values
    y_true = y_true.astype(int).values
    y_pred = y_pred.astype(int)

    unfair_outcomes = 0
    for i in range(len(y_true)):
        js = np.arange(len(y_true))
        
        diff_group = np.abs(prot_col[i]-prot_col[js]) # 1 if diff. group, 0 else
        same_label = 1-np.abs(y_true[i]-y_true[js]) # 1 if same label, 0 else
        diff_outcome = np.abs(y_pred[i]-y_pred[js]) # 1 if diff. outcome, 0 else

        unfair_outcomes += np.sum(diff_group*same_label*diff_outcome)

    return unfair_outcomes / len(y_true) ** 2

def get_score(obj_key, y_pred, y, prot_attr, bounds):
    if obj_key == 'f1':
        score = 1 - f1_score(y, y_pred)
        return score

    elif obj_key == 'ddsp':
        score = statistical_parity_difference(
            y_true=y, 
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        return np.abs(score)

    elif obj_key == 'deod':
        score = average_odds_difference(
            y_true=y, 
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        return np.abs(score)

    elif obj_key == 'deop':
        score = equal_opportunity_difference(
            y_true=y, 
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        return np.abs(score)

    elif obj_key == 'genr':
        within_group = generalized_entropy_error(
            y_true=y,
            y_pred=y_pred
        )
        between_group = between_group_generalized_entropy_error(
            y_true=y,
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        return np.abs(within_group + between_group)
    
    elif obj_key == 'invd':
        score = inverse_distance(
            y_true=y,
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        return score

    elif obj_key == 'comp':
        invd = inverse_distance(
            y_true=y,
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        ddsp = statistical_parity_difference(
            y_true=y, 
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        deod = average_odds_difference(
            y_true=y, 
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        deop = equal_opportunity_difference(
            y_true=y, 
            y_pred=y_pred,
            prot_attr=prot_attr
        )
        fair_vals = np.array([ddsp, deod, deop, invd])
        fair_mins = np.array([
            np.min(bounds['val_ddsp']['min']),
            np.min(bounds['val_deod']['min']),
            np.min(bounds['val_deop']['min']),
            np.min(bounds['val_invd']['min']),
        ])
        fair_maxs = np.array([
            np.max(bounds['val_ddsp']['max']),
            np.max(bounds['val_deod']['max']),
            np.max(bounds['val_deop']['max']),
            np.max(bounds['val_invd']['max']),
        ])

        return np.sum((fair_vals - fair_mins) / (fair_maxs - fair_mins))

    else:
        raise KeyError('objective not supported')
    
def load_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config

def print_config(config):
    for key, value in config.__dict__.items():
        print(f'{key}: {value}')

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_search_space_size(model_key):
    cs = get_config_space(model_key, seed=1)
    hparams = cs.get_hyperparameter_names()
    
    return len(hparams)

def save_data(dataset_dir, seed, X, y, s, split_type):
    X.to_csv(f'{dataset_dir}/X_{split_type}_{seed}.csv', index=False)
    np.savetxt(f'{dataset_dir}/y_{split_type}_{seed}.csv', y, delimiter=',')
    np.savetxt(f'{dataset_dir}/s_{split_type}_{seed}.csv', s, delimiter=',')

def load_data(dataset_dir, seed, split_type):
    X = pd.read_csv(f'{dataset_dir}/X_{split_type}_{seed}.csv')
    y = np.genfromtxt(f'{dataset_dir}/y_{split_type}_{seed}.csv', delimiter=',')
    s = np.genfromtxt(f'{dataset_dir}/s_{split_type}_{seed}.csv', delimiter=',')
    
    return X, y, s

def get_strat(y, s):
    return np.char.add(y.astype(str), s.astype(str))

def get_protected_attribute(dataset_key):
    protected_map = {
        'adult': 'race',
        'compas': 'race',
        'german': 'sex',
        'bank': 'age',
        'lawschool': 'race'
    }

    return protected_map[dataset_key]

def get_objective_names(objective_name):
    objective_map = {
        'f1': ['f1'],
        'f1_ddsp': ['f1', 'ddsp'],
        'f1_deod': ['f1', 'deod'],
        'f1_deop': ['f1', 'deop'],
        'f1_invd': ['f1', 'invd'],
        'f1_multi': ['f1', 'ddsp', 'deod', 'deop', 'invd'],
        'f1_comp': ['f1', 'ddsp', 'deod', 'deop', 'invd', 'comp'],
    }

    return objective_map[objective_name]

def get_fairness_objective_settings(objective_name):
    return {
        'name': f"val_{objective_name}",
        'threshold': 1.0,
        'lower_is_better': True,
        'normalize': {'algorithm': 'NoOPScaler'}
    }

def get_n_max_gen(model_name, objective_name):
    n_max_gen_map = {
        "f1": {
            "rf": 36,
            "nn": 38,
            "xgb": 36,
            "hgb": 36
        },
        "f1_ddsp": {
            "rf": 38,
            "nn": 40,
            "xgb": 38,
            "hgb": 38
        },
        "f1_deop": {
            "rf": 38,
            "nn": 40,
            "xgb": 38,
            "hgb": 38
        },
        "f1_deod": {
            "rf": 38,
            "nn": 40,
            "xgb": 38,
            "hgb": 38
        },
        "f1_invd": {
            "rf": 38,
            "nn": 40,
            "xgb": 38,
            "hgb": 38
        },
        "f1_multi": {
            "rf": 40,
            "nn": 42,
            "xgb": 40,
            "hgb": 40,
        },
        "f1_comp": {
            "rf": 40,
            "nn": 42,
            "xgb": 40,
            "hgb": 40,
        }
    }

    return n_max_gen_map[objective_name][model_name]

def get_pop_size(model_name, objective_name):
    pop_size_map = {
        "f1": {
            "rf": 26,
            "nn": 28,
            "xgb": 26,
            "hgb": 26

        },
        "f1_ddsp": {
            "rf": 28,
            "nn": 30,
            "xgb": 28,
            "hgb": 28

        },
        "f1_deop": {
            "rf": 28,
            "nn": 30,
            "xgb": 28,
            "hgb": 28
        },
        "f1_deod": {
            "rf": 28,
            "nn": 30,
            "xgb": 28,
            "hgb": 28
        },
        "f1_invd": {
            "rf": 28,
            "nn": 30,
            "xgb": 28,
            "hgb": 28
        },
        "f1_multi": {
            "rf": 36,
            "nn": 38,
            "xgb": 36,
            "hgb": 36
        },
        "f1_comp": {
            "rf": 40,
            "nn": 42,
            "xgb": 40,
            "hgb": 40,
        }
    }

    return pop_size_map[objective_name][model_name] 

def calc_hypervolume(observations_df, obj_keys):
    ref_point = [1] * len(obj_keys)

    test_keys = [f"{key}_test" for key in obj_keys]
    data = observations_df[obj_keys].values
    test_data = observations_df[test_keys].values

    result = {
        'hypervolume': [0.0],
        'hypervolume_test': [0.0],
    }
    for i in range(1, len(observations_df)):
        result['hypervolume'].append(hypervolume(data[:i]).compute(ref_point))
        result['hypervolume_test'].append(hypervolume(test_data[:i][is_pareto_front(data[:i])]).compute(ref_point))

    return pd.DataFrame(result)

def get_experiments(obj_names, model_names, dataset_names, seeds=10):
    test_keys, val_keys = ['test_f1', 'test_ddsp', 'test_deod', 'test_deop', 'test_invd', 'test_comp'], ['val_f1', 'val_ddsp', 'val_deod', 'val_deop', 'val_invd', 'val_comp']

    bounds = {dataset: {val_key: {'min': None, 'max': None} for val_key in val_keys[1:5]} for dataset in dataset_names}
    mo_experiments = {}
    for k, obj_name in enumerate(obj_names):

        opt_name = 'NSGA_II_DEFAULT'
        if obj_name == 'f1':
            opt_name = 'GA_DEFAULT'
        elif obj_name in ('f1_multi', 'f1_comp'):
            opt_name = 'NSGA_III_DEFAULT'
        
        for j, dataset in enumerate(dataset_names):

            mo_experiments[dataset] = {'function_values': pd.DataFrame(columns=val_keys+test_keys)}

            for i, model in enumerate(model_names):
  
                logger.info(f'Compiling /work/dlclarge2/robertsj-fairmohpo/{opt_name}/{obj_name}/{model}/{dataset} seeds')

                hparams = get_config_space(model, seed=0).get_hyperparameter_names()

                mo_experiments[(obj_name, model, dataset)] = {}
                mo_experiments[(obj_name, model, dataset)]['function_values'] = pd.DataFrame(columns=val_keys+test_keys)
                mo_experiments[(obj_name, model, dataset)]['archive'] = pd.DataFrame(columns=hparams)
                for seed in range(seeds):

                    mo_experiments[(obj_name, model, dataset)][str(seed)] = {}
    
                    try:
                        experiment = load_experiment(f"/work/dlclarge2/robertsj-fairmohpo/{opt_name}/{obj_name}/{model}/{dataset}/{seed}/fairmohpo.pickle")
                        ind_completed_runs = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
                        trials_data = experiment.fetch_trials_data(trial_indices=ind_completed_runs).df
                        rows = pd.DataFrame({key: trials_data[trials_data['metric_name'] == key]['mean'].values for key in val_keys+test_keys})
                    except:
                        print(f"{obj_name}/{model}/{dataset}/{seed} failed")
                        continue
                        
                    arms = experiment.arms_by_name
        
                    archive = {hparam: [] for hparam in hparams}
                    for arm in range(len(arms)):
                        for hparam in hparams:
                            archive[hparam].append(arms[str(arm)].parameters[hparam])

                    mo_experiments[(obj_name, model, dataset)]['archive'] = mo_experiments[(obj_name, model, dataset)]['archive'].append(pd.DataFrame(archive))
                    mo_experiments[(obj_name, model, dataset)][str(seed)]['observations'] = rows
                    mo_experiments[(obj_name, model, dataset)]['function_values'] = mo_experiments[(obj_name, model, dataset)]['function_values'].append(rows)
                    
                    mo_experiments[dataset]['function_values'] = mo_experiments[dataset]['function_values'].append(rows)

            if obj_name in ('f1_ddsp', 'f1_deod', 'f1_deop', 'f1_invd'):
                bounds[dataset][f'val_{obj_name.split("_")[1]}']['min'] = np.min(mo_experiments[dataset]['function_values'][[f'val_{obj_name.split("_")[1]}', f'test_{obj_name.split("_")[1]}']])
                bounds[dataset][f'val_{obj_name.split("_")[1]}']['max'] = np.max(mo_experiments[dataset]['function_values'][[f'val_{obj_name.split("_")[1]}', f'test_{obj_name.split("_")[1]}']])

    return mo_experiments, bounds

def normalize_value(x, X):
    X = list(X)
    X.append(x)
    X_norm = normalize_array(X)
    return X_norm[-1]

    # return (x-np.min(X))/(np.max(X)-np.min(X))

def normalize_array(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))


# Matplotlib radar chart code

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta