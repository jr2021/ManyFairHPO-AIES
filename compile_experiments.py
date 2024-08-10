from utils import get_experiments
import pickle as pkl

model_names, dataset_names = ['xgb', 'rf', 'nn'], ['german', 'compas', 'bank', 'adult', 'lawschool']
seeds = 10

experiments, bounds = get_experiments(
    obj_names=['f1_comp'], # 'f1_ddsp', 'f1_deod', 'f1_deop', 'f1_invd', 'f1_multi'
    model_names=model_names,
    dataset_names=dataset_names,
)

with open('/work/dlclarge2/robertsj-fairmohpo/experiments_comp.pkl', 'wb') as f:
    pkl.dump(experiments, f)

# with open('/work/dlclarge2/robertsj-fairmohpo/bounds.pkl', 'wb') as f:
    # pkl.dump(bounds, f)