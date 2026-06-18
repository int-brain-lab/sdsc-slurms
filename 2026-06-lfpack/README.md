# LFPack compression — SDSC Popeye

Compresses the IBL LFP recordings into HDF5 archives using SVD + wavelet-packet thresholding
(`lfpack`).  Produces two compression levels per PID (default and aggressive).

## Output layout

```
/mnt/home/owinter/ceph/lfpack/
  <pid>/
    lf_compressed.h5            default  (ε=150, α=28)
    lf_compressed_aggressive.h5 aggressive (ε=450, α=96)
    compress.done               sentinel written on success
    <pid>_compress.error        traceback written on failure (absent on success)
```

## Commands

```bash
# First run (or add new PIDs)
cd /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack && sbatch compress.sbatch

# Re-run all PIDs from scratch
cd /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack && sbatch compress.sbatch --overwrite

# Check progress
grep -c 'compress.done' /mnt/home/owinter/ceph/lfpack/*/compress.done 2>/dev/null | tail -1
ls /mnt/home/owinter/ceph/lfpack/*/compress.done | wc -l

# Check errors
ls /mnt/home/owinter/ceph/lfpack/*/*.error 2>/dev/null
```

## Parallelism strategy

The node has 48 cores and 1 TB of local NVMe (`/scratch`).  Each PID goes through two
sequential stages that are both CPU-bound and already internally parallelised:

1. **Cadzow decimation** — ProcessPoolExecutor, reads the `.cbin`, writes a float32
   checkpoint (~1.4 GB at 250 Hz × 384 channels × 1 h).
2. **SVD + WP compression** (×2 levels) — joblib Parallel, reads the checkpoint,
   writes two tiny H5 archives (~2 MB each at CR ≈ 500–1500×).

The checkpoint is written to `/scratch` (fast local NVMe) and shared between the default
and aggressive passes, so the expensive Cadzow step runs **only once per PID**.

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

**Throughput estimate (~1000 PIDs / 1000 hours):**

| Stage        | Time / PID (12 cores) |
|--------------|-----------------------|
| Cadzow       | ~8 min                |
| SVD+WP (×2)  | ~2 min                |
| Total        | ~10 min               |

With 4 parallel PIDs: ~4 PIDs / 10 min → **~24 PIDs/h → ~42 h for 1000 PIDs**.
Fits in two back-to-back 24 h jobs.  The `compress.done` sentinel skips completed
PIDs on the second run.

## Sync results to local / Elbocal

```bash
rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/lfpack /mnt/s0/Data/lfpack
```