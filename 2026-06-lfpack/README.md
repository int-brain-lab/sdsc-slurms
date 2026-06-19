# LFPack compression — SDSC Popeye

Compresses the IBL LFP recordings into HDF5 archives using SVD + wavelet-packet thresholding
(`lfpack`).  Produces two compression levels per PID (default and aggressive).

## Output layout

```
/mnt/home/owinter/ceph/ea/cells/
  <pid>/
    lf_compressed.h5              default   (ε=150, α=28)
    lf_compressed_aggressive.h5   aggressive (ε=450, α=96)
    lf_resampled_car_cadzow.npy   Cadzow checkpoint archived after first run (~1.4 GB)
    <pid>_compress.error          traceback written on failure (absent on success)
```

`lf_compressed_aggressive.h5` is written last and acts as the completion sentinel.

## Commands

```bash
# First run (or add new PIDs)
cd /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack && sbatch compress.sbatch

# Re-run all PIDs from scratch
cd /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack && sbatch compress.sbatch --overwrite

# Check progress — PIDs fully done (both H5 files present)
find /mnt/home/owinter/ceph/ea/cells -maxdepth 2 -name 'lf_compressed_aggressive.h5' | wc -l

# Check progress — default pass only
find /mnt/home/owinter/ceph/ea/cells -maxdepth 2 -name 'lf_compressed.h5' | wc -l

# Check errors
ls /mnt/home/owinter/ceph/ea/cells/*/*.error 2>/dev/null

# Live job stats (replace JOBID)
sstat -j <JOBID>.batch --format=JobID,AveCPU,MaxRSS,MaxVMSize,NTasks
sacct -j <JOBID> --format=JobID,Elapsed,CPUTime,CPUTimeRAW,NCPUS
```

## Parallelism strategy

The node has 48 cores and 1 TB of local NVMe (`/scratch`).  Each PID goes through two
sequential stages that are both CPU-bound and already internally parallelised:

1. **Cadzow decimation** — ProcessPoolExecutor, reads the `.cbin`, writes a float32
   checkpoint (~1.4 GB at 250 Hz × 384 channels × 1 h) to **local NVMe scratch**.
   The checkpoint is then archived to ceph and reused on subsequent runs.
2. **SVD + WP compression** (×2 levels) — joblib Parallel, reads the checkpoint,
   writes two tiny H5 archives (~2 MB each, CR ≈ 250–300×).

```
N_OUTER = 4  (PIDs in parallel via joblib outer loop)
N_INNER = 12 (cores per PID for both Cadzow and SVD+WP stages)
N_OUTER × N_INNER = 48  ← fully utilises the node
```

**Why 4 × 12 and not, say, 1 × 48?**

- The Cadzow ProcessPoolExecutor uses a `spawn` context.  Scaling beyond ~12–16 workers
  per PID gives diminishing returns due to process-spawn overhead and the relatively
  small number of FFT-optimal chunks (~1400 per hour of recording).
- Running 4 PIDs in parallel hides I/O latency: while one PID is downloading/reading
  its `.cbin` from ceph, three others are computing.
- Scratch usage stays negligible: 4 × 1.4 GB ≈ 6 GB at any moment, well within 1 TB.

**Measured throughput (job 2441369, 17 h 51 min wall-clock, 100% CPU efficiency):**

118 PIDs completed in 17 h 51 min → **6.6 PIDs/h**, ~36 min/PID average with 4 parallel workers.
The per-PID time splits as:

| Scenario                          | Time / PID |
|-----------------------------------|------------|
| Fresh (no Cadzow cache)           | ~65 min    |
| Cached Cadzow on ceph (most PIDs) | ~36 min    |

With the scratch fix, Cadzow is written to NVMe and archived to ceph on first run,
so subsequent jobs reuse the cache via a fast sequential copy and stay near ~36 min/PID.

A single 48-core node processes ~160 PIDs per 24 h job.
This exceeds the wall-time limit for large datasets; use a SLURM array to scale horizontally.

## Horizontal scaling with SLURM arrays

Split the PID list across N array tasks, each running on its own 48-core node:

```bash
#SBATCH --array=0-9   # 10 jobs × 3.7 PIDs/h ≈ 37 PIDs/h → ~27 h for 1000 PIDs
```

Slice `pids` in `compress.py` by `$SLURM_ARRAY_TASK_ID` before the joblib call.
The H5 existence check (both files present → skip) makes reruns and overlapping arrays safe.

## Sync results to local / Elbocal

```bash
rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cells /mnt/s0/Data/lfpack
```