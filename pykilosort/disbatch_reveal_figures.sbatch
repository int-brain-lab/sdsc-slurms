#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="reveal"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH -o /mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests/reveal.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

PYTHON_EXEC=/mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/python
PYTHON_SCRIPT=~/Documents/PYTHON/sdsc-slurms/pykilosort/reveal_figures.py

module load python/3.10.10

$PYTHON_EXEC $PYTHON_SCRIPT