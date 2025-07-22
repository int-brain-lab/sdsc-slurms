#!/bin/bash

pid=$2
start_time=$8
duration=${10}

module purge


source /mnt/home/prai1/projects/passive_ephys/.venv/bin/activate

pid_dir=/mnt/sdceph/users/prai1/data/projects/psychedlics/logs/${pid}
#Create a pid directory if it doesn't exist
mkdir -p ${pid_dir}


python /mnt/home/prai1/projects/psychedlics/computation.py "$@" > ${pid_dir}/${start_time}_${duration}.log 2>&1