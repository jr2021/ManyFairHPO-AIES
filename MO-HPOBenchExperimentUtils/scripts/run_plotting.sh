n_seeds=1
models=(rf xgb nn)
optimizer=NSGA_II_DEFAULT
datasets=(adult german compas)
config_dir=configs
obj_names=(f1 f1_ddsp f1_deod f1_deod f1_genr f1_multi)

rm -rf slurm
mkdir slurm

for obj_name in "${obj_names[@]}"
do
    for dataset_name in "${datasets[@]}"
    do
        for model_name in "${models[@]}"
        do
            for seed in $(seq $n_seeds)
            do  
                echo $obj_name $dataset_name $model_name $seed 
                python run_plot_script.py --output_path /home/robertsj/FairMOHPO/MO-HPOBenchExperimentUtils/experiments/exp_hpobench/logs/$optimizer/$obj_name/$model_name/$dataset_name/$seed --result_dir /home/robertsj/FairMOHPO/MO-HPOBenchExperimentUtils/experiments/exp_hpobench/logs/$optimizer/$obj_name/$model_name/$dataset_name/$seed --use_only_benchmark fairmohpo_$model_name
            done
        done
    done
done
