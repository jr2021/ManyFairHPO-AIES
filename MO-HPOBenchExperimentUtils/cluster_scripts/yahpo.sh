#!/bin/sh
#SBATCH -p mlhiwidlc_gpu-rtx2080 #partition
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-24:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -a [1] # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /work/dlclarge1/sharmaa-mohpo # Change working_dir
#SBATCH -o /work/dlclarge1/sharmaa-mohpo/logs/plot.%A.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge1/sharmaa-mohpo/logs/plot.%A.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sharmaa@informatik.uni-freiburg.de
# Activate virtual env so that run_experiment can load the correct packages

cd $(ws_find mohpo)
cd MO-HPOBenchExperimentUtils
pwd
export PATH="/home/muelleph/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate mo_hpobench_39


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
                           python3  scripts/plot_mo_metric.py --output_path '/work/dlclarge1/sharmaa-mohpo/plots/YAHPOGymMOBenchmark/yahpo_surrogate_iaml_iaml_xgboost_41146' --is_surrogate 'True' --x_limit 12 --x_label 'mmce' --y_label 'nf' --bench_name 'YAHPOGymMOBenchmark' --title 'yahpo_surrogate_iaml_xgboost_41146' --path '/work/dlclarge1/sharmaa-mohpo/ResultsYAHPO/yahpo_surrogate_iaml_iaml_xgboost_41146'
                           exit $?
fi
if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
                           python3  scripts/plot_mo_metric.py --output_path '/work/dlclarge1/sharmaa-mohpo/plots/YAHPOGymMOBenchmark/yahpo_surrogate_iaml_iaml_xgboost_40981' --is_surrogate 'True' --x_limit 12 --x_label 'mmce' --y_label 'nf' --bench_name 'YAHPOGymMOBenchmark' --title 'yahpo_surrogate_iaml_xgboost_40981' --path '/work/dlclarge1/sharmaa-mohpo/ResultsYAHPO/yahpo_surrogate_iaml_iaml_xgboost_40981'
                           exit $?
fi
if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
                           python3  scripts/plot_mo_metric.py --output_path '/work/dlclarge1/sharmaa-mohpo/plots/YAHPOGymMOBenchmark/yahpo_surrogate_iaml_iaml_super_41146' --is_surrogate 'True' --x_limit 12 --x_label 'mmce' --y_label 'nf' --bench_name 'YAHPOGymMOBenchmark' --title 'yahpo_surrogate_iaml_super_41146' --path '/work/dlclarge1/sharmaa-mohpo/ResultsYAHPO/yahpo_surrogate_iaml_iaml_super_41146'
                           exit $?
fi
if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
                           python3  scripts/plot_mo_metric.py --output_path '/work/dlclarge1/sharmaa-mohpo/plots/YAHPOGymMOBenchmark/yahpo_surrogate_iaml_iaml_super_40981' --is_surrogate 'True' --x_limit 12 --x_label 'mmce' --y_label 'nf' --bench_name 'YAHPOGymMOBenchmark' --title 'yahpo_surrogate_iaml_super_40981' --path '/work/dlclarge1/sharmaa-mohpo/ResultsYAHPO/yahpo_surrogate_iaml_iaml_super_40981'
                           exit $?
fi
if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
                           python3  scripts/plot_mo_metric.py --output_path '/work/dlclarge1/sharmaa-mohpo/plots/YAHPOGymMOBenchmark/yahpo_surrogate_iaml_iaml_ranger_40981' --is_surrogate 'True' --x_limit 12 --x_label 'mmce' --y_label 'nf' --bench_name 'YAHPOGymMOBenchmark' --title 'yahpo_surrogate_iaml_iaml_ranger_40981' --path '/work/dlclarge1/sharmaa-mohpo/ResultsYAHPO/yahpo_surrogate_iaml_iaml_ranger_40981'
                           exit $?
fi
if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
                           python3  scripts/plot_mo_metric.py --output_path '/work/dlclarge1/sharmaa-mohpo/plots/YAHPOGymMOBenchmark/yahpo_surrogate_iaml_iaml_ranger_41146' --is_surrogate 'True' --x_limit 12 --x_label 'mmce' --y_label 'nf' --bench_name 'YAHPOGymMOBenchmark' --title 'yahpo_surrogate_iaml_iaml_ranger_41146' --path '/work/dlclarge1/sharmaa-mohpo/ResultsYAHPO/yahpo_surrogate_iaml_iaml_ranger_41146'
                           exit $?
fi
