echo "SOURCE ENV VARIABLES"
echo ""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#export NUMEXPR_MAX_THREADS=${SLURM_JOB_CPUS_PER_NODE}  # IMPORTANT!!
export NUMEXPR_MAX_THREADS=1  # IMPORTANT!!

# Hopefully this solves a memory leak
# https://github.com/pytorch/pytorch/issues/41486#issuecomment-716849467
export MKL_DISABLE_FAST_MM=1

# important for Dask to not fail on large cluster setups
export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=10
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=200
export DASK_DISTRIBUTED__COMM__RETRY__COUNT=4
export DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT=30
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=
export DASK_DISTRIBUTED__ADMIN__TICK__LIMIT=1800s
