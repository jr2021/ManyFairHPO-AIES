# MO-HPOBenchExperimentUtils
Experiment scripts for MO-HPOBench


## Installation:

We recommend to create an environment with `conda`.

### VARIABLES
```bash 
EMAIL_ADDR=ph_mueller@posteo.de
ENV_NAME=mo_hpobench_39
WS_NAME=mo_hpobench
```

### Set the correct proxies. 
```
export HTTP_PROXY=http://tfproxy.informatik.uni-freiburg.de:8080
export HTTPS_PROXY=http://tfproxy.informatik.uni-freiburg.de:8080
git config --global http.proxy $HTTP_PROXY
git config --global https.proxy $HTTPS_PROXY
export http_proxy=http://tfproxy.informatik.uni-freiburg.de:8080
export https_proxy=http://tfproxy.informatik.uni-freiburg.de:8080
```

### Create Environment And Workspace
```
ws_allocate ${WS_NAME} 180 -r 14 -m ${EMAIL_ADDR}
conda update -n base -c defaults conda
conda create -n ${ENV_NAME} python=3.9
```

### Installation
```
cd $(ws_find ${WS_NAME})
conda activate ${ENV_NAME}

conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pygmo==2.16.1

git clone https://github.com/automl/MO-HPOBenchExperimentUtils
cd MO-HPOBenchExperimentUtils
git checkout hpobench
pip install -e .
cd .. 

git clone https://github.com/automl/HPOBench
cd HPOBench
git checkout development
pip install -e .
cd ..
```


### HPOBench
Clone from github/automl/HPOBench

### Pygmo
If you are using `python >=3.9`, you have to install pygmo with `conda`, because it is currently not on pypi. 
```bash 
conda install pygmo==2.6.1
```

### MO_HYPERBAND

Please download the mo_hyperband implementation in the same 
directory as the MO-Experiment Utils: 
```text
dir/
    MO-HPOBenchExperimentUtils/
    HPOBench/
    mo_hyperband/
```

Use the following command to install mo_hyperband
``` bash
git clone git@github.com:ayushi-3536/mo_hyperband.git
pip install pygmo~=2.6
pip install -r ./mo_hyperband/requirements.txt
```

### Install MO_ExpUtils
```bash 
cd MO-HPOBenchExperimentUtils
pip install -e .
```

#### Additional info:

##### Get Bounds for Nasbench benchmarks. 
```text 
CIFAR 10 VALID:
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
```