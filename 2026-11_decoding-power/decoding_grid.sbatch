#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="decgrid"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-11_decoding-power/_log_.%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

# BWM stimulus-side decoding grid for the neuron-pool power analysis (ibl-ai-agent project
# repeated-site-decoding-power, script 005). Resumable: finished (level, region, session) files are skipped.
# Single node:      sbatch decoding_grid.sbatch
# Up to 4 nodes:    sbatch --array=0-3 decoding_grid.sbatch   (tasks striped over SLURM_ARRAY_TASK_ID)

IBL_AGENT=/mnt/home/owinter/Documents/ibl-ai-agent
PYTHON_EXEC=$IBL_AGENT/.venv/bin/python
PYTHON_SCRIPT=$IBL_AGENT/projects/repeated-site-decoding-power/exploratory-analyses/005_decoding_grid.py

module load python/3.11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # liblinear is single-threaded; one process per core

cd $IBL_AGENT  # data_locations.local.yaml is resolved from the repo root
$PYTHON_EXEC -W ignore -u $PYTHON_SCRIPT
