#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="sync_lag_decoding"
#SBATCH --array=0-2
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=700G
#SBATCH --time=23:59:59
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-11_probe-sync-lag/_log_.%x_%A_%a.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-11_probe-sync-lag/_log_.%x_%A_%a.err

# 544 tasks (17 sessions x 4 targets x 2 conditions x 4 blocks of 50 pseudo-sessions) spread over 3 full nodes (account limit),
# 46 single-threaded worker processes per node. Re-submitting only runs the tasks without a decoding/_done marker.
PYTHON_EXEC=/mnt/home/owinter/Documents/sync-lag/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-11_probe-sync-lag/decoding.py

module load python/3.11
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

$PYTHON_EXEC -u $PYTHON_SCRIPT --node $SLURM_ARRAY_TASK_ID --n-nodes $SLURM_ARRAY_TASK_COUNT --workers 46
