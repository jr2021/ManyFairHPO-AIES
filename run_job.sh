#!/bin/bash

#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --time 1-0:0:0
#SBATCH --output slurm/%x-%A.out
#SBATCH --error slurm/%x-%A.err
#SBATCH --mem 6GB
#SBATCH --nodes 1
#SBATCH --ntasks 1

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

optimizer=$1
obj_name=$2
model_name=$3
dataset_name=$4
seed=${SLURM_ARRAY_TASK_ID}

start=`date +%s`

python -u runner.py --objective_name $obj_name --model_name $model_name --dataset_name $dataset_name --seed $seed --optimizer_name $optimizer

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime