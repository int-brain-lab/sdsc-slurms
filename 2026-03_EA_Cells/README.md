## Main commands
Output directory, one folder per pid:
`cd /mnt/home/owinter/ceph/ea/cells`

Output directory for aggregate tables
`cd /mnt/home/owinter/ceph/ea/cell_aggregates`

Code directory on Popeye:
`cd /mnt/home/owinter/Documents/sdsc-slurms/2026-03_EA_Cells`
`cd /mnt/home/owinter/Documents/ephys-atlas`


# 24cb326a-e04d-4a18-a96d-1edef60cc40b
Launch the compute
`cd /mnt/home/owinter/Documents/sdsc-slurms/2026-03_EA_Cells && sbatch cells.sbatch`

Re-run all PIDs, overwriting existing HDF5 files:
`cd /mnt/home/owinter/ceph/ea/cells && sbatch cells.sbatch --overwrite`


## Other useful commands (progress check / rsync)

`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cells_aggregates /Users/olivier/Documents/datadisk/paper-ephys-atlas/cells_aggregates`
`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cells_aggregates_f32 /Users/olivier/Documents/datadisk/paper-ephys-atlas/cells_aggregates_f32`

`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cells_aggregates /mnt/s0/Data/paper-ephys-atlas/cells`

Download the results on Mac / Elbocal

`find /mnt/home/owinter/ceph/ea/cells -name 'lf_resampled.bin' | wc -l`



## Output format
Each folder is named by the pid and contains:
    clusters.pqt
    lf_resampled.npy
    stlfp.npy
    stpc.npy
    stpc.png


## Computation timin
```python
[
    ('2026-06-20T21:30:00', 0)
    ('2026-06-21T16:50:00', 213),
    ('2026-06-22T20:30:00', 464),
] 
# 11 per hour
```
