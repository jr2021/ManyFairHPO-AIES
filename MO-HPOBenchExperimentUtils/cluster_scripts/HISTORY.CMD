--max_running_tasks 3 \
--memory_per_job 4000 \
--timelimit 43200 \

-q alldlc_gpu-rtx2080 \
--qos dlc-muelleph \

-q bosch_cpu-cascadelake \



# ------------------------------------------------------------------------------------------------
# YAHPO
# ------------------------------------------------------------------------------------------------
python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q bosch_cpu-cascadelake \
    --timelimit 43700 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh \
    --name yahpo_surrogate \
    --no_mail \
    --array_min 1223 \
    --array_max 1232 \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/yahpo_surrogate \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/yahpo_surrogate \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/yahpo_surrogate.txt




# MO CNN COMBINED 24 H
python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q mlhiwidlc_gpu-rtx2080 \
    --qos dlc-muelleph \
    --timelimit 86400 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh \
    --name rs_mo_cnn \
    --memory_per_job 12000 \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/rs_mo_cnn \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/rs_mo_cnn \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_cnn.txt


# ------------------------------------------------------------------------------------------------
# MO ADULT GPU 24h
# ------------------------------------------------------------------------------------------------
python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q mlhiwidlc_gpu-rtx2080 \
    --qos dlc-muelleph \
    --timelimit 86400 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh \
    --name rs_mo_adult \
    --memory_per_job 12000 \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/mo_adult \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/mo_adult \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_adult.txt

# ------------------------------------------------------------------------------------------------
# NASBENCH 201
# ------------------------------------------------------------------------------------------------
exp_mo_nas_201_cifar100.txt        exp_mo_nas_201_cifar10_valid.txt   exp_mo_nas_201_imagenet_valid.txt
python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q bosch_cpu-cascadelake \
    --timelimit 43700 \
    --mem 12000 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh  \
    --name exp_mo_nas_201_cifar100    \
    --no_mail \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/exp_mo_nas_201_cifar100    \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/exp_mo_nas_201_cifar100     \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_nas_201_cifar100.txt

# ------------------------------------------------------------------------------------------------
# CREATE MISSING FILES
# ------------------------------------------------------------------------------------------------
create_missing_files_exp_mo_adult.txt
create_missing_files_exp_mo_cnn_fashion.txt
create_missing_files_exp_mo_cnn_flower.txt
create_missing_files_exp_mo_nas_201_cifar10_valid.txt
create_missing_files_exp_mo_nas_201_cifar100.txt
create_missing_files_exp_mo_nas_201_imagenet_valid.txt
create_missing_files_yahpo_surrogate_modified.txt


python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q bosch_cpu-cascadelake \
    --timelimit 43700 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh  \
    --name create_missing_files_yahpo_surrogate_modified    \
    --no_mail     \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/create_missing_files_yahpo_surrogate_modified   \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/create_missing_files_yahpo_surrogate_modified   \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_yahpo_surrogate_modified.txt


# ------------------------------------------------------------------------------------------------
# REDUCED SEARCH SPACE FAIR fgrrm 7592
# ------------------------------------------------------------------------------------------------
python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q bosch_cpu-cascadelake \
    --timelimit 43700 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh  \
    --name fair_reduced_cs  \
    --no_mail     \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/fair_reduced_cs   \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/fair_reduced_cs     \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/yahpo_surrogate_only_cont.txt


# ------------------------------------------------------------------------------------------------
# Create Run Stats
# ------------------------------------------------------------------------------------------------
python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q bosch_cpu-cascadelake \
    --timelimit 43700 \
    --mem 24000 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh  \
    --name run_stats  \
    --no_mail  \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/collect_run_stats   \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/collect_run_stats     \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/cmd_run_stats.txt


# ------------------------------------------------------------------------------------------------
# Create Plots
# ------------------------------------------------------------------------------------------------
(yahpo_surrogate_iaml_iaml_xgboost_41146|yahpo_surrogate_iaml_iaml_xgboost_40981|yahpo_surrogate_iaml_iaml_super_41146|yahpo_surrogate_iaml_iaml_super_40981|yahpo_surrogate_iaml_iaml_ranger_41146|yahpo_surrogate_iaml_iaml_ranger_40981|yahpo_surrogate_fair_fair_xgboost_7592|yahpo_surrogate_fair_fair_super_317599|yahpo_surrogate_fair_fair_super_14965|yahpo_surrogate_fair_fair_ranger_317599|yahpo_surrogate_fair_fair_ranger_14965|yahpo_surrogate_fair_fair_fgrrm_7592)

(yahpo_surrogate_iaml_iaml_xgboost_41146|
yahpo_surrogate_iaml_iaml_xgboost_40981|
yahpo_surrogate_iaml_iaml_super_41146|
yahpo_surrogate_iaml_iaml_super_40981|
yahpo_surrogate_iaml_iaml_ranger_41146|
yahpo_surrogate_iaml_iaml_ranger_40981|
yahpo_surrogate_fair_fair_xgboost_7592|
yahpo_surrogate_fair_fair_super_317599|
yahpo_surrogate_fair_fair_super_14965|
yahpo_surrogate_fair_fair_ranger_317599|
yahpo_surrogate_fair_fair_ranger_14965|
yahpo_surrogate_fair_fair_fgrrm_7592)

python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/slurm_helper.py \
    -q bosch_cpu-cascadelake \
    --timelimit 43700 \
    --mem 24000 \
    --startup /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/cluster_scripts/startup.sh  \
    --name plotting  \
    --no_mail  \
    --array_min 34 \
    --dependency_afterany 13850945 \
    -o /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/plots   \
    -l /work/dlclarge2/muelleph-mo_hpobench/OPT_LOG/plots     \
    /work/dlclarge2/muelleph-mo_hpobench/CMDs/plot_cmds.txt