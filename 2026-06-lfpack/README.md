# LFPack compression — SDSC Popeye

Compresses IBL LFP recordings into HDF5 archives with `lfpack` (SVD + wavelet-packet
thresholding). ADC-saturated stretches are detected on the raw LFP band and muted before
compression; the saturated intervals are stored in each archive. Three tiers are written
per PID (small, default, fine); see the release table at the end for the versions.

## Output layout

```
$OUTPUT_ROOT/                     # /mnt/home/owinter/ceph/ea/denoised_lfp (override via env)
  <pid>/
    lf_compressed_v05_a14_small.h5     small   (ε=100, α=14)
    lf_compressed_v05_a07_default.h5   default (ε=100, α=7)
    lf_compressed_v05_a2p5_fine.h5     fine    (ε=100, α=2.5)
    lf_resampled_car_cadzow_v05.npy    Cadzow checkpoint, archived once every tier is done (~1.4 GB)
    channel_labels.npy                 bad-channel labels from detect_bad_channels.py
    lf_compressed_v0{3,4}_*.h5         earlier releases, left in place
    <pid>_compress.error               traceback on failure (absent on success)
```

Each tier is written via atomic rename; a PID is done when all three exist. The checkpoint
name carries the release, because a checkpoint is reused on reruns: v05 never reads the v04
checkpoint (`lf_resampled_car_cadzow.npy`, old Cadzow) and deletes it when the PID starts.
A resubmitted job therefore resumes where it stopped; `--overwrite` is only needed to redo
finished v05 PIDs.

Only the first tier computes Cadzow; the other two reuse its checkpoint. `compress.py`
copies its Cadzow settings into their `processing` provenance attr, which lfpack leaves
empty (`"cadzow": null`) for a reused checkpoint.

## Commands

```bash
cd ~/Documents/sdsc-slurms/2026-06-lfpack
PY=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python

# Full run (skips PIDs already complete)
sbatch compress.sbatch

# Validate a few PIDs into a fresh folder (interactive, no queue)
OUTPUT_ROOT=~/ceph/ea/denoised_lfp $PY -u compress.py --pids <pid1> <pid2>   # or --limit 3

# Reprocess everything after a pipeline change: point at a new empty folder and run
# normal mode (fresh folder → no stale caches → muting always applies), then delete
# the old folder once happy.
sbatch --export=ALL,OUTPUT_ROOT=~/ceph/ea/denoised_lfp_muted compress.sbatch

# Force-recompute in place (deletes H5s + Cadzow checkpoint)
sbatch compress.sbatch --overwrite

# Progress / errors
find $OUTPUT_ROOT -maxdepth 2 -name 'lf_resampled_car_cadzow_v05.npy' | wc -l   # PIDs done
ls $OUTPUT_ROOT/*/*.error 2>/dev/null                                       # failures
sacct -j <JOBID> --format=JobID,Elapsed,CPUTime,NCPUS
```

## Parallelism

The node has 48 cores and node-local scratch storage for the working directory
(`compress.sbatch`'s `SCRATCH_BASE`, currently `/tmp` -- was `/scratch` until an
SDSC-side config change 2026-09-02 made it permission-denied; capacity/type of
`/tmp` on these nodes hasn't been characterized, worth checking if a run starts
failing with disk-space errors instead). Each PID runs two internally parallel
stages: Cadzow decimation (writes the ~1.4 GB checkpoint to scratch, archived to
ceph and reused on reruns) then SVD+WP compression (one pass per `PARAMS` tier,
~2 MB each). The job runs
`N_OUTER=4` PIDs concurrently × `N_INNER=12` cores = 48. Four in parallel hides ceph I/O
latency; scaling one PID past ~12–16 workers gives diminishing returns.

**v05 Cadzow is slower**: ~14–18 min per NP1 PID on 12 cores, ~3.4 core-hours/PID in all
(jobs 2504932, 2504938), i.e. ~12 PIDs/h/node; the 3-node sweep of ~1,100 PIDs takes ~30 h.
The v04 figures below predate it.

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

`reduce.py` and `reduce.py --bwm` write the merged archives at the root of `OUTPUT_ROOT`. Copy
them into one local folder per cohort. Expect ~5–9 MB/s, i.e. a few hours per release. `--partial`
keeps a dropped transfer and macOS `openrsync` resumes it with `--append` (it has no
`--append-verify`), so check the md5 against ceph afterwards.

```bash
OUTPUT_ROOT=ceph/ea/denoised_lfp
LOCAL=/Users/olivier/Documents/datadisk/lfp-processing/lfpack
rsync -a --partial --append "popeye:$OUTPUT_ROOT/lf_compressed_v05_*_all.h5"     $LOCAL/v05/
rsync -a --partial --append "popeye:$OUTPUT_ROOT/lf_compressed_v05_*_all_bwm.h5" $LOCAL/v05_bwm/
ssh popeye "md5sum $OUTPUT_ROOT/lf_compressed_v05_*.h5"; md5sum $LOCAL/v05*/*.h5
```

Label each folder separately (`--local-root $LOCAL/v05`, then `$LOCAL/v05_bwm`, see below):
`attach_ibl_metadata.py` writes into every `*compressed*.h5` under the root, so keep other
versions out of it. Once labelled, the local archives are the release; the ceph copies stay
unlabelled.

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

## Publish on S3

Upload the **labelled** local archives. BWM goes to the public bucket, at the root of
`ibl-agent-data/` next to the v03 BWM files (not a `bwm/` subfolder). The full ephys-atlas
cohort goes to the private bucket.

```bash
LOCAL=/Users/olivier/Documents/datadisk/lfp-processing/lfpack
# BWM (699 PIDs, public): lf_compressed_v05_{a14_small,a07_default,a2p5_fine}_all_bwm.h5
aws --profile ibl s3 sync $LOCAL/v05_bwm s3://ibl-brain-wide-map-public/resources/ibl-agent-data \
  --exclude '*' --include 'lf_compressed_v05_*_all_bwm.h5' --dryrun
# ephys-atlas (1099 PIDs, private): lf_compressed_v05_{a14_small,a07_default,a2p5_fine}_all.h5
aws --profile ibl s3 sync $LOCAL/v05 s3://ibl-brain-wide-map-private/resources/lfp/ephys-atlas \
  --exclude '*' --include 'lf_compressed_v05_*_all.h5' --dryrun
```

Drop `--dryrun` to upload. From the laptop that took ~5.5 h for the 133 GB of v04.

| release | BWM (public) | ephys-atlas (private) |
|---|---|---|
| v05 (in progress; lfpack 1.2.0, ibl-neuropixel ≥ 1.14 Cadzow: full grid + shrinkage, NP1/NP2 routing, bad channels estimated by Cadzow; codec as v04) | — | — |
| v04 (2026-10-01; lfpack 1.0, ε = 100, m = 32, α = 14 / 7 / 2.5) | small 7.7 · default 14.9 · fine 31.9 GB | small 11.2 · default 21.5 · fine 46.1 GB |
| v03 (deprecated) | mild 23.3 · default 15.6 · aggressive 8.1 GB | mild 34.2 · default 23.0 · aggressive 12.1 GB, in `aggregates/atlas/projects/ibl_neuropixel_brainwide_01/lfp_aggregates/` (the path `ephysatlas.data` reads) |

Release notes, one line per change of the data (rationale and benchmarks in oliche-quarto `analyses/`):

- **v05** — Cadzow only: removes the CSD seams at the Cadzow window boundaries and halves the
  bad-channel lines; sizes per tier within ±5 %, SNR against the denoised input −2 dB
  (`analyses/2026-10-lfpack-v05`). Archives record their provenance (`LFPackReader.provenance`).
  1,097 of 1,099 PIDs: 4037a618-e6ba-4142-a6d2-3e32ef25fd1b and 46630957-63fd-4d0d-a2ba-8766f0796db8 have
  all 384 channels labelled bad, which hangs ibl-neuropixel 1.14 `_fill_grid`; their v04 archives
  were empty (~110 kB, CR > 10,000).
- **v04** — codec: lfpack 1.0 format 2, shared spatial basis m = 32, ε = 100, tiers set by α
  (`analyses/2026-06-lfp-compression-svd`).
- **v03** — default / aggressive tiers (ε = 150, α = 28 / ε = 450, α = 96),
  then mild (ε = 100, α = 14); saturation muting.
