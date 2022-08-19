#!/bin/bash

# Requirements
n_tasks=19
n_cpus=1
mem="4G"

# Locations
in_dir="/mnt/sdceph/users/ibl/data"
out_dir="/mnt/sdceph/users/ibl/data/resources/compressed_rep_site_julia"
working_dir="/mnt/home/jhuntenburg/slurm_jobs/lossy_compression"
task_file="$working_dir/tasks.disbatch"

mkdir -p $working_dir/logs

# Load FI modules and source virtualenv
module purge
module load disBatch/2.0
module load fftw/3.3.10
module load python/3.8.12
source /mnt/home/jhuntenburg/Documents/PYTHON/envs/compressenv/bin/activate

# Create task file with common prefix (functions, cpus, out_dir) but different in_files and logs
paths=(
  "mrsicflogellab/Subjects/SWC_038/2020-07-29/001/raw_ephys_data/probe01"
  "zadorlab/Subjects/CSH_ZAD_026/2020-09-04/001/raw_ephys_data/probe00"
  "churchlandlab/Subjects/CSHL052/2020-02-21/001/raw_ephys_data/probe00"
  "danlab/Subjects/DY_010/2020-01-27/001/raw_ephys_data/probe00"
  "zadorlab/Subjects/CSH_ZAD_019/2020-08-14/001/raw_ephys_data/probe00"
  "wittenlab/Subjects/ibl_witten_25/2020-12-07/002/raw_ephys_data/probe01"
  "zadorlab/Subjects/CSH_ZAD_011/2020-03-22/001/raw_ephys_data/probe00"
  "cortexlab/Subjects/KS020/2020-02-07/001/raw_ephys_data/probe01"
  "wittenlab/Subjects/ibl_witten_29/2021-06-08/001/raw_ephys_data/probe00"
  "hoferlab/Subjects/SWC_014/2019-12-10/001/raw_ephys_data/probe01"
  "cortexlab/Subjects/KS023/2019-12-10/001/raw_ephys_data/probe01"
  "mainenlab/Subjects/ZFM-02370/2021-04-29/001/raw_ephys_data/probe00"
  "angelakilab/Subjects/NYU-47/2021-06-21/003/raw_ephys_data/probe00"
  "churchlandlab_ucla/Subjects/UCLA011/2021-07-20/001/raw_ephys_data/probe00"
  "hoferlab/Subjects/SWC_043/2020-09-16/002/raw_ephys_data/probe01"
  "danlab/Subjects/DY_013/2020-03-12/001/raw_ephys_data/probe00"
  "zadorlab/Subjects/CSH_ZAD_001/2020-01-14/002/raw_ephys_data/probe00"
  "mrsicflogellab/Subjects/SWC_054/2020-10-05/001/raw_ephys_data/probe01"
  "danlab/Subjects/DY_016/2020-09-12/001/raw_ephys_data/probe00"
)

echo "#DISBATCH PREFIX /usr/bin/time -f '%M %E' python run_lossy_compression.py $out_dir " > $task_file

i=0
for path in "${paths[@]}"
do
    echo "$in_dir/$path > logs/task_${i}.log 2>&1" >> $task_file
    i=$((i+1))
done 

# Submit the slurm job, run ntasks at a time, each with n_pus cores
sbatch --partition=gen --ntasks=$n_tasks --cpus-per-task=$n_cpus --mem-per-cpu=$mem disBatch $task_file
