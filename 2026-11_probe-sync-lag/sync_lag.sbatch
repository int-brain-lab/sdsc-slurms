#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="sync_lag"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --time=11:59:59
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-11_probe-sync-lag/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-11_probe-sync-lag/_log_.%x_%j.err

PYTHON_EXEC=/mnt/home/owinter/Documents/sync-lag/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-11_probe-sync-lag/sync_lag.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT $@  # --step {extract,wheel,aggregate} [--overwrite] [--pids ...]
