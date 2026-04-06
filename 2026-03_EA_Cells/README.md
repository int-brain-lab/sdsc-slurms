## Main commands
Output directory, one folder per pid:
`/mnt/home/owinter/ceph/ea/cells`

Output directory for aggregate tables
`/mnt/home/owinter/ceph/ea/cell_aggregates`

Code directory on Popeye:
`/mnt/home/owinter/Documents/sdsc-slurms/2026-03_EA_Cells`
`/mnt/home/owinter/Documents/ephys-atlas`


# 24cb326a-e04d-4a18-a96d-1edef60cc40b
Launch the compute
`sbatch cells.sbatch`


## Other useful commands (progress check / rsync)

`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cell_aggregates /Users/olivier/Documents/datadisk/paper-ephys-atlas/popeye`
`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cell_aggregates /mnt/s0/Data/paper-ephys-atlas/cells`

Download the results on Mac / Elbocal

`find /mnt/home/owinter/ceph/ea/cells -name 'lf_resampled.bin' | wc -l`



## Output format
Each folder is named by the pid and contains:
    clusters.pqt
    lf_resampled.npy
    stlfp.npy
    stpc.npy
    stpc.png
