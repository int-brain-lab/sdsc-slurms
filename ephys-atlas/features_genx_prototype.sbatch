#!/usr/bin/bash
#SBATCH -p genx
#SBATCH --job-name="features"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/ephys-atlas/_log_.%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/ephys-atlas/features.py

module load python/3.11

$PYTHON_EXEC $PYTHON_SCRIPT

