#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --job-name="${PID}_spikesorting"
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=${TIME}
#SBATCH -o ${LOG_PATH}/${PID}_spikesorting.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cal2254@columbia.edu

PYTHON_EXEC=/mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/python
PYTHON_SCRIPT=~/Documents/PYTHON/sdsc-slurms/pykilosort/pykilosort_task.py
export SDSC_PATCH_PATH=/mnt/sdceph/users/ibl/data/quarantine/tasks_${USER}/

module load python/3.10.10
module load cuda/11.8.0

$PYTHON_EXEC $PYTHON_SCRIPT ${PID}
