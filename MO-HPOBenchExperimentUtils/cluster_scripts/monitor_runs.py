import pandas as pd
from pathlib import Path
from MOHPOBenchExperimentUtils.utils.hpobench_utils import HPOBenchMetrics
from typing import Union
from time import sleep


def monitor_used_wallclock_time(result_path: Union[str, Path],
                                output_path: Union[str, Path],
                                output_file_name: str = 'current_progress.csv'):
    print('RESULT PATH: ', result_path)

    result_path = Path(result_path)
    output_path = Path(output_path)
    output_file = output_path / output_file_name

    csv_files = list(result_path.rglob('*_df.csv'))
    print(f'Found {len(csv_files)} files')

    exceptions = {}
    runs = []
    for csv_file in csv_files:

        run_id = csv_file.parent.name
        optimizer = csv_file.parent.parent.name
        experiment = csv_file.parent.parent.parent.name

        entry = {'optimizer': optimizer, 'run_id': run_id, 'experiment': experiment}

        try:
            sleep(0.01)
            with open(csv_file, 'r') as fh:
                df = pd.read_csv(fh)
            run_info = {
                'num_configs': df.trial_index.max(),
                'total_cost': df.loc[df.metric_name == HPOBenchMetrics.COST, 'mean'].sum() / 3600
            }
        except Exception as e:
            exceptions[(experiment, optimizer, run_id, csv_file)] = e
            run_info = {
                'num_configs': 0,
                'total_cost': 0
            }

        entry = {**entry, **run_info}
        runs.append(entry)

    all_runs = pd.DataFrame(runs) \
        .sort_values(['experiment', 'optimizer', 'run_id'])
    all_runs = all_runs.reindex(columns=['optimizer', 'experiment', 'run_id', 'num_configs', 'total_cost'])

    print(f'Write Monitoring CSV to {output_file}')
    all_runs.to_csv(output_file)
    print(f'Done')

    from pprint import pprint
    pprint(exceptions)

    print()
    print()

try:
    result_path = Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results')
    output_path = Path('/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results')
    monitor_used_wallclock_time(result_path=result_path, output_path=output_path)
except:
    pass

try:
    monitor_used_wallclock_time(
        result_path=Path('/work/dlclarge2/muelleph-mo_hpobench/Results'),
        output_path=Path('/work/dlclarge2/muelleph-mo_hpobench/'),
        output_file_name='progress_results.csv'
    )
except:
    pass

try:
    monitor_used_wallclock_time(
        result_path=Path('/work/dlclarge2/muelleph-mo_hpobench/Results2'),
        output_path=Path('/work/dlclarge2/muelleph-mo_hpobench/'),
        output_file_name='progress_results2.csv'
    )
except:
    pass
