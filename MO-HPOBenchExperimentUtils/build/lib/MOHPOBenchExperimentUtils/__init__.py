import logging
import sys
from MOHPOBenchExperimentUtils.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
from MOHPOBenchExperimentUtils.utils.experiment_utils import save_experiment, save_experiment_df, load_experiment
from MOHPOBenchExperimentUtils.utils.pareto import pareto, nDS, computeHV2D, nDS_index, crowdingDist, \
    contributionsHV3D, computeHV
from MOHPOBenchExperimentUtils.utils import *
from ax.core.experiment import logger as ax_exp_logger

from loguru import logger
logger.remove(0)
logger.add(
    # colorize=False,
    # format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
    format='{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - <level>{message}</level>',
    level='INFO',
    enqueue=True,
    sink=sys.stdout,
)
logger.info('Import MOHPOBenchExperimentUtils.')

ax_exp_logger.setLevel(logging.WARNING)
