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

## Attach IBL metadata (on popeye)

`reduce.py` and `reduce.py --bwm` write the merged archives at the root of `OUTPUT_ROOT`.
`attach_ibl_metadata.py` then labels them **in place on ceph** (`attach.sbatch`), so the
labelled ceph archives are the release and nothing is synced to a laptop for this step. It
attaches what the compression job doesn't have: per-channel brain locations
(`ml`/`ap`/`dv`/`atlas_id`/`acronym` on `<pid>/00/meta`, from the ephys-atlas features
dataframe with a `SpikeSortingLoader` fallback) and the registered sample↔time sync knots on
every scale. Data access goes through `OneSdsc` (local ceph files); the features table is
downloaded once to `~/data/ephys-atlas/features`. Lookups are offline: ONE local mode on its
full cache tables (refreshed at the start of each run, one download) plus `eid`/`pname` from
the probe-details parquet, ~1.2 s/PID (~20 min for the cohort). Remote ONE costs ~8 Alyx
requests per PID (~6 s/PID, rate limited); a PID whose offline lookup fails is retried
through Alyx, and the run reports how many needed it. It labels every `--pattern` archive
directly under `--root` (default `lf_compressed_v05_*_all*.h5`, both cohorts, all tiers).

Before writing channel annotations it runs a **geometry sanity check**: the source's
on-probe `lateral_um`/`axial_um` are matched by exact position against the archive's
`geometry_x`/`geometry_y` (both µm). If too few channels match, the source is the wrong
probe — those PIDs are reported and their channels skipped (sync still attaches).

```bash
cd ~/Documents/sdsc-slurms/2026-06-lfpack
sbatch --dependency=afterok:<reduce job> attach.sbatch            # write attrs in place
sbatch attach.sbatch --dry-run                                    # report only
```

Re-running `reduce.py` rewrites the merged archives without the labels: attach again after it.

## Publish on S3

Upload the **labelled** archives straight from ceph with `publish.sbatch` (AWS CLI in
`~/.local/bin`, `ibl` profile in `~/.aws` on popeye). BWM goes to the public bucket, at the
root of `ibl-agent-data/` next to the v03 BWM files (not a `bwm/` subfolder); the full
ephys-atlas cohort goes to the private bucket. Set `VERSION` in the script. Earlier releases
stay on S3 next to the new one, to reproduce past results (`s3 sync` without `--delete` never
removes files).

```bash
cd ~/Documents/sdsc-slurms/2026-06-lfpack
sbatch publish.sbatch --dryrun                                  # list what would be uploaded
sbatch --dependency=afterok:<attach job> publish.sbatch         # upload once attach is done
```

A local copy is no longer part of the release; to inspect archives on the laptop, rsync them
from `popeye:ceph/ea/denoised_lfp/` (`--partial --append`, then compare md5 sums).

| release | BWM (public) | ephys-atlas (private) |
|---|---|---|
| v05 (2026-11-09; lfpack 1.2.0, ibl-neuropixel 1.14.0 Cadzow: full grid + shrinkage, NP1/NP2 routing, bad channels estimated by Cadzow; codec as v04) | small 10.1 · default 19.9 · fine 43.2 GB (697 PIDs) | small 15.0 · default 29.7 · fine 64.2 GB (1097 PIDs; CR 130 / 66 / 30) |
| v04 (2026-10-01; lfpack 1.0, ε = 100, m = 32, α = 14 / 7 / 2.5) | small 7.7 · default 14.9 · fine 31.9 GB | small 11.2 · default 21.5 · fine 46.1 GB |
| v03 (deprecated) | mild 23.3 · default 15.6 · aggressive 8.1 GB | mild 34.2 · default 23.0 · aggressive 12.1 GB, in `aggregates/atlas/projects/ibl_neuropixel_brainwide_01/lfp_aggregates/` (the path `ephysatlas.data` reads) |

Release notes, one line per change of the data (rationale and benchmarks in oliche-quarto `analyses/`):

- **v05** — Cadzow only: removes the CSD seams at the Cadzow window boundaries and halves the
  bad-channel lines; sizes per tier within ±5 %, SNR against the denoised input −2 dB
  (`analyses/2026-10-lfpack-v05`). Archives record their provenance (`LFPackReader.provenance`).
  1,097 of 1,099 PIDs: 4037a618-e6ba-4142-a6d2-3e32ef25fd1b and 46630957-63fd-4d0d-a2ba-8766f0796db8 have
  all 384 channels labelled bad, which hangs ibl-neuropixel 1.14 `_fill_grid`; their v04 archives
  were empty (~110 kB, CR > 10,000). No sync knots for 1e176f17, 4b93a168, 701026df, eedba3b6
  (no sync dataset on Alyx). bf96f6d6 had no `channel_labels.npy` (detection crashed): its
  `labels` attr was copied from the small tier into default/fine. Uploaded 2026-11-09 from the
  popeye login node (`bash publish.sbatch` in tmux, ~1.5 h, sizes checked against ceph).
- **v04** — codec: lfpack 1.0 format 2, shared spatial basis m = 32, ε = 100, tiers set by α
  (`analyses/2026-06-lfp-compression-svd`).
- **v03** — default / aggressive tiers (ε = 150, α = 28 / ε = 450, α = 96),
  then mild (ε = 100, α = 14); saturation muting.
