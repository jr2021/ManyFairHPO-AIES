from pathlib import Path


def generate_nasbench_files(
    code_dir: Path, result_dir: Path, cmd_dir: Path, optimizer: str
):
    benchmarks = [
        'mo_nas_201_cifar10_valid', 'mo_nas_201_cifar100', 'mo_nas_201_imagenet_valid',
        'mo_nas_1shot1_1', 'mo_nas_1shot1_2', 'mo_nas_1shot1_3',
        'mo_nas_101_A', 'mo_nas_101_B', 'mo_nas_101_C'
    ]

    for experiment_name in benchmarks:
        generate_command_file(
            code_dir, result_dir, cmd_dir, experiment_name, optimizer,
            append=True, cmd_file_name='nasbench.txt'
        )


def generate_yahpo_surrogate_files(
    code_dir: Path, result_dir: Path, cmd_dir: Path, run_file: str, optimizer_arg: str,
    use_cont_only: bool,
    cmd_file_name: str = 'yahpo_surrogate.txt',
):

    mapping_yahpo_iaml_small = {
        'iaml_glmnet': [1489, 40981],
        'iaml_rpart': [1489, 41146],
        'iaml_ranger': [40981, 41146],
        'iaml_xgboost': [40981, 41146],
        'iaml_super': [40981, 41146],
    }

    mapping_yahpo_fair_small = {
        'fair_fgrrm': [7592, 14965],
        'fair_rpart': [7592, 317599],
        'fair_ranger': [14965, 317599],
        'fair_xgboost': [7592, 317599],
        'fair_super': [14965, 317599],
    }

    mapping_yahpo_fair_small_cont_only = {
        'fair_fgrrm': [7592], # 14965],
        # 'fair_rpart': [7592, 317599],
        # 'fair_ranger': [14965, 317599],
        # 'fair_xgboost': [7592, 317599],
        # 'fair_super': [14965, 317599],
    }

    for (experiment_base, mapping) in [
            ('yahpo_surrogate_iaml', mapping_yahpo_iaml_small),
            ('yahpo_surrogate_fair', mapping_yahpo_fair_small),
            ('yahpo_surrogate_fair_cont_only', mapping_yahpo_fair_small_cont_only),
        ]:

        is_cont_only_benchmark = 'cont_only' in experiment_base

        if (not use_cont_only and is_cont_only_benchmark) \
            or (use_cont_only and not is_cont_only_benchmark):
            continue

        for scenario, instances in mapping.items():
            for instance in instances:
                experiment_name = f'{experiment_base}_{scenario}_{instance}'
                generate_command_file(
                    code_dir=code_dir, result_dir=result_dir, cmd_dir=cmd_dir,
                    experiment_name=experiment_name, run_file=run_file, optimizer_arg=optimizer_arg,
                    append=True,
                    cmd_file_name=cmd_file_name
                )

def generate_command_file(
        code_dir: Path, result_dir: Path, cmd_dir: Path,
        experiment_name: str,
        run_file: str,
        optimizer_arg: str = None,
        append: bool = False,
        cmd_file_name: str = None,
):

    opt_file = code_dir / 'experiments/exp_hpobench' / f'{run_file}.py'

    cmd_dir.mkdir(exist_ok=True, parents=True)
    if cmd_file_name is None:
        cmd_file = cmd_dir / f'exp_{experiment_name}_opt_{optimizer_arg}.txt'
    else:
        cmd_file = cmd_dir / cmd_file_name

    file_opt = 'w' if not append else 'a'
    with cmd_file.open(file_opt) as fh:
        if append:
            fh.write('\n')

        for run_id in range(10):
            command = f'python {opt_file} ' \
                  f'--experiment_name {experiment_name} ' \
                  f'--optimizer_name {optimizer_arg} ' \
                  f'--run_id {run_id} ' \
                  f'--output_path {result_dir}'
            fh.write(command)
            fh.write('\n')

    print(f'Finished writing to {cmd_file}')


# generate_nasbench_files(
#     code_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils'),
#     result_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/Results'),
#     cmd_dir=Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results'),
#     optimizer='random_search',
# )

# Generate the Small Experiemnts for the YAHPO IAML Benchmarks
# generate_yahpo_surrogate_files(
#     code_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils'),
#     result_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/ResultsYAHPO'),
#     # cmd_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/CMDs'),
#     cmd_dir=Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results'),
#     run_file='random_search',
# )

# Example for the flower benchmark
# generate_command_file(
#     code_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils'),
#     cmd_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/CMDs'),
#     result_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/Results'),
#     optimizer='random_search',
#     experiment_name='mo_cnn_fashion'
# )

MAP_OPTIMIZER_RUNFILE = {
    'random_search': ['RANDOM_SEARCH'],
    'pymoo_experiments': ['NSGA_III_DEFAULT', 'MOEA_D_DEFAULT', 'AGE_MOEA_DEFAULT'],
    'mo_hb': ['MO_HB_RANDOM_WEIGHTS', 'MO_HB_PAREGO', 'MO_HB_GOLOVIN', 'MO_HB_NSGA-II', 'MO_HB_EPSNET'],
    'generic_experiments': ['MO_SMAC_GP', 'MO_SMAC_PAREGO', 'MO_BANANAS', 'MOSH_BANANAS'],
}

GROUP_OPTIMIZER = {
    'sf_only_cont_wo_conditions': ['NSGA_III_DEFAULT', 'MOEA_D_DEFAULT', 'AGE_MOEA_DEFAULT'],
    'sf_only_cont_w_conditions': [],
    'sf_only_numeric_w_conditions': ['MO_BANANAS', ],
    'sf_mixed_wo_conditions': [],
    'sf_mixed_w_conditions': ['MO_SMAC_GP', 'MO_SMAC_PAREGO', 'RANDOM_SEARCH'],
    'mf_only_cont_wo_conditions': [],
    'mf_only_cont_w_conditions': [],
    'mf_only_numeric_wo_conditions': [],
    'mf_only_numeric_w_conditions': ['MOSH_BANANAS'],
    'mf_mixed_wo_conditions': [],
    'mf_mixed_w_conditions': ['MO_HB_RANDOM_WEIGHTS', 'MO_HB_PAREGO', 'MO_HB_GOLOVIN', 'MO_HB_NSGA-II', 'MO_HB_EPSNET'],
}

all_optimizers = []
for optimizers in GROUP_OPTIMIZER.values():
    all_optimizers.extend(optimizers)
GROUP_OPTIMIZER['all'] = all_optimizers


MAP_BENCHMARK_OPTIMIZER = {
    'mo_adult': GROUP_OPTIMIZER['sf_mixed_w_conditions']
              + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
              + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
              + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_cnn_flower': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                   + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                   + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                   + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_cnn_fashion': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                    + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                    + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                    + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_201_cifar10_valid': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                              + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                              + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                              + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_201_cifar100': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                         + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                         + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                         + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_201_imagenet_valid': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                               + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                               + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                               + GROUP_OPTIMIZER['mf_mixed_w_conditions'],
    
    'mo_nas_101_A': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                  + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                  + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                  + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_101_B': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                  + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                  + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                  + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_101_C': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                  + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                  + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                  + GROUP_OPTIMIZER['mf_mixed_w_conditions'],
    
    'mo_nas_1shot1_CS_1': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                        + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                        + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                        + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_1shot1_CS_2': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                        + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                        + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                        + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'mo_nas_1shot1_CS_3': GROUP_OPTIMIZER['sf_mixed_w_conditions']
                        + GROUP_OPTIMIZER['sf_only_numeric_w_conditions']
                        + GROUP_OPTIMIZER['mf_only_numeric_w_conditions']
                        + GROUP_OPTIMIZER['mf_mixed_w_conditions'],

    'yahpo': GROUP_OPTIMIZER['mf_mixed_w_conditions']
           + GROUP_OPTIMIZER['sf_mixed_w_conditions'],

    'yahpo_only_cont': GROUP_OPTIMIZER['all']
}

# for benchmark in [k for k in MAP_BENCHMARK_OPTIMIZER.keys()]:
for benchmark in [
    # 'mo_adult',
    # 'mo_cnn_flower',
    # 'mo_cnn_fashion',
    
    # 'mo_nas_101_A',
    # 'mo_nas_101_B',
    # 'mo_nas_101_C',
    #
    # 'mo_nas_1shot1_CS_1',
    # 'mo_nas_1shot1_CS_2',
    # 'mo_nas_1shot1_CS_3',
    #
    # 'mo_nas_201_cifar10_valid',
    # 'mo_nas_201_cifar100',
    # 'mo_nas_201_imagenet_valid',

    # 'yahpo',
    'yahpo_only_cont'
]:
    default_settings = dict(
        # cmd_dir=Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results/CMDS'),
        cmd_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/CMDs'),
        # code_dir=Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils/'),
        code_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils'),
        # result_dir=Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results/Test123'),
        result_dir=Path('/work/dlclarge2/muelleph-mo_hpobench/ResultsTest_19_7'),
    )

    cmd_files = {
        'yahpo': 'yahpo_surrogate.txt',
        'yahpo_only_cont': 'yahpo_surrogate_only_cont.txt',
    }
    cmd_file_name = cmd_files.get(benchmark, f'exp_{benchmark}.txt')
    if (default_settings['cmd_dir'] / cmd_file_name).exists():
        (default_settings['cmd_dir'] / cmd_file_name).unlink()

    for optimizer_arg in MAP_BENCHMARK_OPTIMIZER[benchmark]:
        run_file = [
            run_file for run_file, optimizers in MAP_OPTIMIZER_RUNFILE.items()
            if optimizer_arg in optimizers
        ]
        assert len(run_file) == 1
        common_settings = dict(
            optimizer_arg=optimizer_arg,
            run_file=run_file[0],
            cmd_file_name=cmd_file_name,
            **default_settings,
        )

        if benchmark in ['yahpo', 'yahpo_only_cont']:
            # Generate the Small Experiemnts for the YAHPO IAML Benchmarks
            generate_yahpo_surrogate_files(
                use_cont_only=not benchmark == 'yahpo',
                **common_settings,
            )

        else:
            generate_command_file(
                experiment_name=benchmark,
                cmd_file_name=f'exp_{benchmark}.txt',
                append=True,
                **common_settings,
            )
