#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpack"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --time=3-00:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%A_%a.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%A_%a.err

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-06-lfpack/compress.py

# Node-local scratch base. Was /scratch (local NVMe); switched to /tmp 2026-09-02
# after an SDSC-side config change left /scratch permission-denied on compute
# nodes (pcn-9-07, pcn-9-16 both confirmed). Single named constant so a future
# path change is a one-line edit.
SCRATCH_BASE=/tmp
export SCRATCH_ROOT=${SCRATCH_BASE}/lfpack_${SLURM_JOB_ID}
mkdir -p $SCRATCH_ROOT

# Pin BLAS/OpenMP to one thread per process: the job already saturates the node with
# N_OUTER × N_INNER = 48 worker processes, so letting each also spawn 48 BLAS threads
# (numpy SVD, scipy decimate/FFT) would oversubscribe the 48 cores ~48×.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT $@  # pass --overwrite to recompute all
PYTHON_EXIT=$?

rm -rf $SCRATCH_ROOT
exit $PYTHON_EXIT  # propagate the real exit code -- the rm above always succeeds
                    # even on missing paths, so without this sacct's ExitCode
                    # silently reports 0:0 regardless of whether python crashed