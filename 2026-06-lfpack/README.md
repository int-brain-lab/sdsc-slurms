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

Throughput (measured job 2451554, 6 PIDs, `--overwrite`, with the BLAS thread caps):
**~2 core-hours/PID** — the parallelism-independent invariant to plan from. At 4×12 with
full waves (~90% CPU efficiency) that is **~20 PIDs/h/node**; budget ~15/h to absorb
ceph-fetch on first touch, warmup, and the half-empty final wave. Per-PID cost scales with
recording length. (Short jobs show lower CPU efficiency — e.g. the 6-PID validation ran at
56%, since its second wave filled only 2 of 4 slots; full sweeps stay near 90%.)

## Horizontal scaling with SLURM arrays

`compress.sbatch` sets `--array`; each task slices `pids` by `$SLURM_ARRAY_TASK_ID`, and
the sentinel-skip makes reruns and overlapping arrays safe. At ~15–20 PIDs/h/node a single
node clears ~360–480 PIDs within the 24 h wall-limit, so size the array to keep each task
under that:

```bash
#SBATCH --array=0-3   #  4 nodes → ~60–80 PIDs/h → ~500 PIDs in  ~7–8 h
#SBATCH --array=0-9   # 10 nodes → ~150–200 PIDs/h → ~700 PIDs in ~4 h
```

Get the total PID count from the first line each task logs — `Task 0/N: queuing M PIDs`
(multiply M by the array size) — then divide by ~15–20/h/node.

## Sync results to local

```bash
rsync -av --progress -e ssh --include='*/' --include='lf_compressed*.h5' --exclude='*' \
  popeye:$OUTPUT_ROOT /Users/olivier/Documents/datadisk/lfp-processing/lfpack
```

## Local post-processing: attach IBL metadata

`attach_ibl_metadata.py` runs **locally** (not on the cluster) against the synced archives
and attaches IBL information the cluster job doesn't have: per-channel brain locations
(`ml`/`ap`/`dv`/`atlas_id`/`acronym` on `<pid>/00/meta`, from the ephys-atlas features
dataframe with a `SpikeSortingLoader` fallback) and the sample→time sync affine
(`fs_sync`/`t0_sync` on every scale, fitted from probe sync pulses, QC < 1 ms). It writes
into every `*compressed*.h5` under `--local-root`, for whichever recordings each holds.

Before writing channel annotations it runs a **geometry sanity check**: the source's
on-probe `lateral_um`/`axial_um` are compared channel-by-channel against the archive's
stored `geometry_x`/`geometry_y` (both µm). A mismatch means the source channel order
doesn't line up with the archive, so brain locations would be misassigned — those PIDs are
reported and their channels skipped (sync still attaches).

```bash
python attach_ibl_metadata.py --dry-run   # report only: geometry check + failure summary
python attach_ibl_metadata.py             # write attrs into the local archives
```
