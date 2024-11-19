#!/usr/bin/bash
#SBATCH -p genx
#SBATCH --job-name="features"
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/ephys-atlas/_log_.features.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

PYTHON_EXEC=/mnt/home/owinter/Documents/PYTHON/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/ephys-atlas/features.py

module load python/3.11

$PYTHON_EXEC $PYTHON_SCRIPT

