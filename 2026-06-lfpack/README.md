# LFPack compression — SDSC Popeye

Compresses IBL LFP recordings into HDF5 archives with `lfpack` (SVD + wavelet-packet
thresholding). ADC-saturated stretches are detected on the raw LFP band and muted before
compression; the saturated intervals are stored in each archive. Two levels are written
per PID (default and aggressive).

## Output layout

```
$OUTPUT_ROOT/                     # /mnt/home/owinter/ceph/ea/denoised_lfp (override via env)
  <pid>/
    lf_compressed.h5              default    (ε=150, α=28)
    lf_compressed_aggressive.h5   aggressive (ε=450, α=96)   ← completion sentinel
    lf_resampled_car_cadzow.npy   Cadzow checkpoint archived after first run (~1.4 GB)
    <pid>_compress.error          traceback on failure (absent on success)
```

`lf_compressed_aggressive.h5` is written last via atomic rename, so its presence = done.

## Commands

```bash
cd ~/Documents/sdsc-slurms/2026-06-lfpack
PY=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python

# Full run (skips PIDs already complete)
sbatch compress.sbatch

# Validate a few PIDs into a fresh folder (interactive, no queue)
OUTPUT_ROOT=~/ceph/ea/denoised_lfp_test $PY -u compress.py --pids <pid1> <pid2>   # or --limit 3

# Reprocess everything after a pipeline change: point at a new empty folder and run
# normal mode (fresh folder → no stale caches → muting always applies), then delete
# the old folder once happy.
sbatch --export=ALL,OUTPUT_ROOT=~/ceph/ea/denoised_lfp_muted compress.sbatch

# Force-recompute in place (deletes H5s + Cadzow checkpoint)
sbatch compress.sbatch --overwrite

# Progress / errors
find $OUTPUT_ROOT -maxdepth 2 -name 'lf_compressed_aggressive.h5' | wc -l   # PIDs done
ls $OUTPUT_ROOT/*/*.error 2>/dev/null                                       # failures
sacct -j <JOBID> --format=JobID,Elapsed,CPUTime,NCPUS
```

## Parallelism

The node has 48 cores and 1 TB local NVMe (`/scratch`). Each PID runs two internally
parallel stages: Cadzow decimation (writes the ~1.4 GB checkpoint to scratch, archived to
ceph and reused on reruns) then SVD+WP compression (×2 levels, ~2 MB each). The job runs
`N_OUTER=4` PIDs concurrently × `N_INNER=12` cores = 48. Four in parallel hides ceph I/O
latency; scaling one PID past ~12–16 workers gives diminishing returns.

Throughput: ~36 min/PID cached, ~65 min fresh → ~160 PIDs / 24 h node.

## Horizontal scaling with SLURM arrays

`compress.sbatch` sets `--array`; each task slices `pids` by `$SLURM_ARRAY_TASK_ID`, and
the sentinel-skip makes reruns and overlapping arrays safe.

```bash
#SBATCH --array=0-9   # 10 nodes → ~37 PIDs/h → ~27 h for 1000 PIDs
```

## Sync results to local

```bash
rsync -av --progress -e ssh --include='*/' --include='lf_compressed*.h5' --exclude='*' \
  popeye:$OUTPUT_ROOT /Users/olivier/Documents/datadisk/lfp-processing/lfpack
```
