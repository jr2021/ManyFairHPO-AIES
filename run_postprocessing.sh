#!/bin/bash

seeds=10
models=(xgb rf nn)
optimizer=DE
datasets=(adult bank german compas)
config_dir=configs
obj_names=(f1_ddsp f1_deod f1_deop)

for obj_name in "${obj_names[@]}"
do
    for dataset_name in "${datasets[@]}"
    do
        for model_name in "${models[@]}"
        do
            sbatch -J post_${model_name} postprocess.sh $optimizer $obj_name $model_name $dataset_name $seeds
        done
    done
done
