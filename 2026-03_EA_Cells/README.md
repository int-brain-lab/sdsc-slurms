
Output directory, one folder per pid:
/mnt/home/owinter/ceph/ea/cells

Directory of tables
/mnt/home/owinter/ceph/ea/cell_aggregates

Code directory on Popeye:
/mnt/home/owinter/Documents/sdsc-slurms/2026-03_EA_Cells

Launch the compute
sbatch cells.sbatch


rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cell_aggregates /Users/olivier/Documents/datadisk/paper-ephys-atlas/popeye