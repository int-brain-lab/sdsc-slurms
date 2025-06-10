#!/bin/bash
#SBATCH --job-name="trials-extraction"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=00:14:00
#SBATCH -o '/mnt/home/mwells/Documents/PYTHON/trials-extraction/popeye_re_extract.out'
#SBATCH -p gen
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=miles.wells@internationalbrainlab.org

export SDSC_HOME='/mnt/home/mwells/Documents/PYTHON/trials-extraction'
export PATH=$PATH:${SDSC_HOME}/Documents/PYTHON/sdsc-slurms
export ONE_SAVE_ON_DELETE=0

${SDSC_HOME}/.venv/bin/python ${SDSC_HOME}/sdsc-slurms/2025-05_trials-extraction/03-re-extract-trials-popeye.py --n-sessions 1000