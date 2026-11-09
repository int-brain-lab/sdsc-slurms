#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpack-attach"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%j.err

# Attach IBL channels + sync metadata in place to the merged archives written by reduce.py.
# Run after reduce.py (e.g. sbatch --dependency=afterok:<reduce job> attach.sbatch); pass
# --dry-run for the geometry check and failure summary only.

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-06-lfpack/attach_ibl_metadata.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT "$@"
