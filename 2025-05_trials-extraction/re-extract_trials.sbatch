#!/bin/bash
#SBATCH --job-name="trials-extraction"
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -o '/mnt/home/clangfield/popeye_re_extract.out'
#SBATCH --open-mode=append


## https://wiki.flatironinstitute.org/SCC/Software/UsingTheGPUNodes

export SDSC_HOME='/mnt/home/clangfield'

${SDSC_HOME}/mambaforge/envs/a/bin/python ${SDSC_HOME}/Documents/PYTHON/ibldevtools/chris/re-extract_trials_popeye_v2.py