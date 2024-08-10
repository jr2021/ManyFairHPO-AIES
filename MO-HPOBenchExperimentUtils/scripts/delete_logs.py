from pathlib import Path
import os
import sys

from loguru import logger


def delete_logs_immediately_stopped(log_directory: Path) -> None:
    # Search for all files:
    log_files = list(log_directory.rglob('*.out'))
    logger.debug(f'Found {len(log_files)} Log Files in {log_directory}')

    count = 0
    for log_file in log_files:
        delete = False

        # Stopped runs should be small (here less than 1MiB)
        if log_file.stat().st_size > 1000000:
            continue

        with open(log_file, 'r') as fh:
            for line in fh.readlines():
                if ('The Directory already exists and is not empty' in line) \
                        or ('No Missing Files.' in line):

                    count += 1
                    delete = True
                    break
        if delete:
            os.remove(log_file)

    logger.debug(f'Deleted: {count} log files')
    logger.info('Done')


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--log_directory', type=str,
        help='/home/lmmista-wap072/Dokumente/Code/MO-HPOBenchExperimentUtils-Results/OPT LOGS'
    )

    args = parser.parse_args()
    log_directory = Path(args.log_directory)
    delete_logs_immediately_stopped(log_directory)
