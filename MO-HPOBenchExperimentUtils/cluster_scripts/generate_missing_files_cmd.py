from pathlib import Path
from typing import Union
from MOHPOBenchExperimentUtils.utils.settings_utils import load_settings
from loguru import logger


def get_run_dirs(result_dir: Union[str, Path]):
    dirs = list(result_dir.rglob(f'*/[0-9]/'))
    assert len(dirs) != 0, f'Found no dirs starting from {result_dir}'

    logger.info(f'Found {len(dirs)} Directories.')
    logger.info(f'Example: {dirs[0]}')
    return dirs


def generate_command_file(cmd_file: str, result_dir: Union[str, Path], recompute_hv=False):
    result_dir = Path(result_dir)
    run_dirs = get_run_dirs(result_dir)
    run_dirs.sort()

    all_settings = load_settings()

    commands = []
    for run_dir in run_dirs:
        exp_name = run_dir.parent.parent.name
        setting = all_settings[exp_name]
        is_surrogate = setting['optimization']['is_surrogate']

        base_commnd = 'python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/scripts/' \
                      'check_and_create_missing_files.py'
        command = base_commnd
        command += f' --run_directory {run_dir}'
        command += ' --is_surrogate' if is_surrogate else ''
        command += ' --recompute_all' if recompute_hv else ''

        commands.append(command)

    with open(cmd_file, 'w') as fh:
        for line in commands:
            fh.write(line + '\n')
    print(f'Finished writing to {cmd_file}')


def generate_command_file_from_cmd_file(old_cmd_file, output_cmd_file, recompute_hv=True):
    old_cmd_file = Path(old_cmd_file)

    with old_cmd_file.open('r') as fh:
        data = fh.readlines()

    all_settings = load_settings()

    base_command = 'python /work/dlclarge2/muelleph-mo_hpobench/MO-HPOBenchExperimentUtils/scripts/check_and_create_missing_files.py ' \
                   '--run_directory {output_path}/{exp_name}/{optimizer_name}/{run_id} ' \
                   '{is_surrogate} ' \
                   '{recompute_all} '

    new_commands = []
    for raw_line in data:
        if raw_line == '\n':
            new_commands.append('')
            continue

        line = raw_line.rstrip('\n')
        split = line.split()
        exp_name = split[3]
        optimizer_name = split[5]
        run_id = split[7]
        output_path = split[9]

        setting = all_settings[exp_name]
        is_surrogate = setting['optimization']['is_surrogate']

        optimizer_name = optimizer_name if optimizer_name != 'RANDOM_SEARCH' else 'random_search'

        new_command = base_command.format(**{
            'output_path': output_path,
            'exp_name': exp_name,
            'optimizer_name': optimizer_name,
            'run_id': run_id,
            'is_surrogate': '--is_surrogate' if is_surrogate else '',
            'recompute_all': '--recompute_all' if recompute_hv else '',
        })

        new_commands.append(new_command)

    with open(output_cmd_file, 'w') as fh:
        for line in new_commands:
            fh.write(line + '\n')
    print(f'Finished writing to {output_cmd_file}')


# generate_command_file(
#     cmd_file='/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_recompute_fair_reduced.txt',
#     result_dir='/work/dlclarge2/muelleph-mo_hpobench/Results_19_7_Fair_ReducedCS',
#     recompute_hv=True,
# )

mappings = [
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/yahpo_surrogate.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_yahpo_surrogate.txt'
     ),
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_adult.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_exp_mo_adult.txt'
    ),
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_cnn_fashion.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_exp_mo_cnn_fashion.txt'
    ),
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_cnn_flower.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_exp_mo_cnn_flower.txt'
    ),
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_nas_201_cifar10_valid.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_exp_mo_nas_201_cifar10_valid.txt'
    ),
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_nas_201_cifar100.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_exp_mo_nas_201_cifar100.txt'
    ),
    (
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/exp_mo_nas_201_imagenet_valid.txt',
        '/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_exp_mo_nas_201_imagenet_valid.txt'
    ),
]
for old_cmd_file, output_cmd_file in mappings:
    print(old_cmd_file)
    generate_command_file_from_cmd_file(
        # old_cmd_file='/work/dlclarge2/muelleph-mo_hpobench/CMDs/yahpo_surrogate_only_cont.txt',
        # output_cmd_file='/work/dlclarge2/muelleph-mo_hpobench/CMDs/create_missing_files_recompute_fair_reduced_from_cmd.txt',
        old_cmd_file=old_cmd_file,
        output_cmd_file=output_cmd_file,
        recompute_hv=False
    )
