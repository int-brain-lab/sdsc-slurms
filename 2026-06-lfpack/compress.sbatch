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

export SCRATCH_ROOT=/scratch/lfpack_${SLURM_JOB_ID}
mkdir -p $SCRATCH_ROOT

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT $@  # pass --overwrite to recompute all

rm -rf $SCRATCH_ROOT