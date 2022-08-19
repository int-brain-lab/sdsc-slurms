#!/bin/bash
# File: launch_slurm.sh
module purge
module load python/3.8.12
source $HOME/Documents/PYTHON/envs/ibllib/bin/activate

# Recommendation: keep scripts in $HOME, and data in ceph
fn_list="$HOME/microphone/microphone.list"
nfiles=$(cat $fn_list | wc -l)

# Launch a Slurm job array with $nfiles entries
echo "sbatch --array=1-$nfiles job.slurm $fn_list"
sbatch --array=1-300 job.slurm $fn_list
