#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="wheel decoding"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=600G
#SBATCH --time=23:59:59
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/bwm/_log_.%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/bwm/wheel_decoding.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT
