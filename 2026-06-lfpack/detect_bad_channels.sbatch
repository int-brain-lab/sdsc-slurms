#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfp_badchan"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --time=1-00:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%A_%a.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%A_%a.err

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-06-lfpack/detect_bad_channels.py

# One BLAS thread per worker process (N_OUTER=8 workers each run detection).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT $@  # pass --overwrite to recompute all
