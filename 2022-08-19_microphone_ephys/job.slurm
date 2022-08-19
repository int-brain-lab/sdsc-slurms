#!/bin/bash
# File: job.slurm

#SBATCH --ntasks=1  # run one thing per job
#SBATCH --cpus-per-task=1
#SBATCH --partition=gen

# the file with the list of files to process
fn_list=$1

# the job array index
# the task ID is automatically set by Slurm
i=$SLURM_ARRAY_TASK_ID

# get the line of the file belonging to this job
# make sure your `sbatch --array=1-X` command uses 1 as the starting index
fn=$(sed -n "${i+1000}p" ${fn_list})

echo "About to process $fn"
python check_microphone.py /mnt/home/owinter/ceph/microphone $fn

