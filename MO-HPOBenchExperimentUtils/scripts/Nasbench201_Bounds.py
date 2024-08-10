# 'name', 'train_acc1es', 'train_acc5es', 'train_losses', 'train_times',
# 'arch_config', 'params', 'flop', 'seed', 'epochs', 'latency',
# 'eval_names', 'eval_acc1es', 'eval_times', 'eval_losses'



import json

# p = '/home/lmmista-wap072/Dokumente/Code/Data_HPOBench/Data/nasbench_201/NAS-Bench-201-v1_1-096897_cifar10-valid.json'
# p = '/home/lmmista-wap072/Dokumente/Code/Data_HPOBench/Data/nasbench_201/NAS-Bench-201-v1_1-096897_ImageNet16-120.json'
p = '/home/lmmista-wap072/Dokumente/Code/Data_HPOBench/Data/nasbench_201/NAS-Bench-201-v1_1-096897_cifar100.json'
with open(p, 'r') as fh:
    data = json.load(fh)

keys = ['777', '888', '999']
min_params = None
max_params = None
min_flop = None
max_flop = None
min_latency = None
max_latency = None

for key in keys:
    configs = list(data[key].keys())
    for config in configs:
        if min_params is None:
            min_params = data[key][config]['params']
        if max_params is None:
            max_params = data[key][config]['params']
        if min_flop is None:
            min_flop = data[key][config]['flop']
        if max_flop is None:
            max_flop = data[key][config]['flop']
        if min_latency is None:
            min_latency = data[key][config]['latency'][0]
        if max_latency is None:
            max_latency = data[key][config]['latency'][0]
        min_params = min(min_params, data[key][config]['params'])
        max_params = max(max_params, data[key][config]['params'])
        min_flop = min(min_flop, data[key][config]['flop'])
        max_flop = max(max_flop, data[key][config]['flop'])
        min_latency = min(min_latency, data[key][config]['latency'][0])
        max_latency = max(max_latency, data[key][config]['latency'][0])

print('min_params ', min_params)
print('max_params ', max_params)
print('min_flop ', min_flop)
print('max_flop ', max_flop)
print('min_latency ', min_latency)
print('max_latency ', max_latency)

"""
CIFAR 10 VALID
min_params  0.073306
max_params  1.531546
min_flop  7.78305
max_flop  220.11969
min_latency  0.007144981308987266
max_latency  0.025579094886779785

CIFAR 100:
min_params  0.079156
max_params  1.537396
min_flop  7.7889
max_flop  220.12554
min_latency  0.007202700332359031
max_latency  0.02614096800486247

IMAGE NET:
min_params  0.080456
max_params  1.538696
min_flop  1.9534
max_flop  55.03756
min_latency  0.005841970443725586
max_latency  0.02822377681732178
"""
