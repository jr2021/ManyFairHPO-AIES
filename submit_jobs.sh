#!/bin/bash

models=(nn rf xgb)
datasets=(german bank compas lawschool adult)

# optimizer=NSGA_II_DEFAULT
# config_dir=configs
# obj_names=(f1_deop f1_invd f1_ddsp f1_deod) 
# for model_name in "${models[@]}"
# do
#     for dataset_name in "${datasets[@]}"
#     do
#         for obj_name in "${obj_names[@]}"
#         do
#             sbatch -J fairmohpo --array=0,1,2,3,4,5,6,7,8,9 run_job.sh $optimizer $obj_name $model_name $dataset_name
#         done
#     done
# done

# optimizer=NSGA_III_DEFAULT
# config_dir=configs
# obj_name=f1_multi

# for model_name in "${models[@]}"
# do
#     for dataset_name in "${datasets[@]}"
#     do
#         sbatch -J fairmohpo --array=0,1,2,3,4,5,6,7,8,9 run_job.sh $optimizer $obj_name $model_name $dataset_name
#     done
# done

optimizer=NSGA_III_DEFAULT
config_dir=configs
obj_name=f1_comp

for model_name in "${models[@]}"
do
    for dataset_name in "${datasets[@]}"
    do
        sbatch -J fairmohpo --array=0,1,2,3,4,5,6,7,8,9 run_job.sh $optimizer $obj_name $model_name $dataset_name
    done
done


# optimizer=GA_DEFAULT
# obj_name=f1
# for model_name in "${models[@]}"
# do
#     for dataset_name in "${datasets[@]}"
#     do
#         sbatch -J ${model_name}_${dataset_name} --array=0,1,2,3,4,5,6,7,8,9 run_job.sh $optimizer $obj_name $model_name $dataset_name
#     done
# done

