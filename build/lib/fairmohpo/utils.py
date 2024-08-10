from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from numpy.random import RandomState
import xgboost as xgb
from aif360.sklearn.datasets import (
    fetch_adult,
    fetch_compas, 
    fetch_german,
    fetch_lawschool_gpa,
    fetch_bank,
)
from aif360.sklearn.metrics import (
    equal_opportunity_difference, 
    generalized_entropy_error,
    average_odds_difference,
    statistical_parity_difference
)
from ConfigSpace import (
    ConfigurationSpace,
    Integer, 
    Float
)
import numpy as np
import yaml
from yaml.loader import SafeLoader

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")


def get_model(config, model_key):
    if model_key == 'rf':
        model = RandomForestClassifier(
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'], 
            min_samples_leaf=config['min_samples_leaf'], 
            max_features=config['max_features'],
            bootstrap=True
        )
        return model
    elif model_key == 'nn':
        depth, width = config['depth'], config['width']
        hidden_layers = [width] * depth
        model = MLPClassifier(
            batch_size=config['batch_size'],
            alpha=config['alpha'],
            learning_rate_init=config['learning_rate_init'],
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            early_stopping=True
        )
        return model
    elif model_key == 'svm':
        model = SVC(
            C=config['C'],
            gamma=config['gamma'],
            probability=True
        )
        return model
    elif model_key == 'xgb':
        model = xgb.XGBClassifier(
            eta=config['eta'],
            max_depth=config['max_depth'],
            colsample_bytree=config['colsample_bytree'],
            reg_lambda=config['reg_lambda'],
            booster='gbtree',
            objective='binary:logistic'
        )
        return model

def get_dataset(dataset_key):

    rng = RandomState(123456789)

    if dataset_key == 'adult':
        X, y, _ = fetch_adult(
            binary_race=True,
            dropna=True,
            subset='test',
            numeric_only=False,
            dropcols=['race', 'education', 'sex']
        )

        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm] 

        # encode target variable
        y = pd.Series(y.factorize(sort=True)[0], index=y.index)
        
        # encode protected attributes
        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
        
        return X, y

    elif dataset_key == 'compas':
        X, y = fetch_compas(
            binary_race=True,
            dropna=True,
            numeric_only=False,
            dropcols=['sex', 'age_cat', 'race', 'c_charge_desc']
        )
    
        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm] 

        # encode protected attributes
        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
       
        # encode target variable and flip b/c recidivism is an unfavourable outcome
        y = 1-pd.Series(y.factorize(sort=True)[0], index=y.index)
        
        return X, y

    elif dataset_key == 'german':

        X, y = fetch_german(
            binary_age=True, 
            dropna=True, 
            numeric_only=False,
            dropcols=['sex', 'age', 'foreign_worker'],
        )

        # shuffle instances
        perm = rng.permutation(len(y))
        X, y = X.iloc[perm], y.iloc[perm] 

        # encode target variable
        y = pd.Series(y.factorize(sort=True)[0], index=y.index)
        
        # encode protected attributes
        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
        
        return X, y

    else:
        raise KeyError('dataset is not supported')

def get_config_space(model_key, seed):
    if model_key == 'rf':
        config_space = ConfigurationSpace(
            name='rf',
            seed=seed,
            space={
                'max_depth': Integer('max_depth', (1, 50), default=10, log=True),
                'min_samples_split': Integer('min_samples_split', (2, 128), default=32, log=True),
                'min_samples_leaf': Integer('min_samples_leaf', (1, 20), default=1),
                'max_features': Float('max_features', (0, 1), default=0.5),
            }
        )
        return config_space
        
    elif model_key == 'nn':
        config_space = ConfigurationSpace(
            name='nn',
            seed=seed,
            space={
                'depth': Integer('depth', (1, 3), default=3),
                'width': Integer('width', (16, 1024), default=64, log=True),
                'batch_size': Integer('batch_size', (4, 256), default=32, log=True),
                'alpha': Float('alpha', (10**-8, 1), default=10**-3, log=True),
                'learning_rate_init': Float('learning_rate_init', (10**-5, 1), default=10**-3, log=True)
            }
        )
        return config_space

    elif model_key == 'svm':
        config_space = ConfigurationSpace(
            name='svm',
            seed=seed,
            space={
                'C': Float('C', (2**-10, 2**10), default=1.0, log=True),
                'gamma': Float('gamma', (2**-10, 2**10), default=1.0, log=True)
            }
        )
        return config_space

    elif model_key == 'xgb':
        config_space = ConfigurationSpace(
            name='xgb',
            seed=seed,
            space={
                'eta': Float('eta', (2**-10, 1.0), default=0.3, log=True),
                'max_depth': Integer('max_depth', (1, 50), default=10, log=True),
                'colsample_bytree': Float('colsample_bytree', (0.1, 1.0), default=1.0),
                'reg_lambda': Float('reg_lambda', (2**-10, 2**10), default=1.0, log=True)
            }
        )
        return config_space

    else:
        raise KeyError('model is not supported')

def get_encoder(X_train):
    pipeline = make_column_transformer(
        (
            OneHotEncoder(
                sparse=False,
                drop='if_binary',
                handle_unknown='ignore'
            ), 
            X_train.dtypes == 'category'
        ),
        remainder=StandardScaler()
    )
    return pipeline

def get_split(X, y, train_idx, val_idx):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    return X_train, y_train, X_val, y_val

def get_sample(model, obj_keys, X, y, prot_attr):
    y_pred = model.predict(X)
    
    sample = np.zeros(len(obj_keys))
    for i, obj_key in enumerate(obj_keys):
         sample[i] = get_score(obj_key, y_pred, y, prot_attr)
    
    return sample

def get_score(obj_key, y_pred, y, prot_attr):
    if obj_key == 'f1':
        f1 = f1_score(y.values, y_pred)
        return 1 - f1

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
        score = generalized_entropy_error(
            y_true=y,
            y_pred=y_pred
        )
        return np.abs(score)
        
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
        'adult': 'sex',
        'compas': 'race',
        'german': 'sex'
    }

    return protected_map[dataset_key]