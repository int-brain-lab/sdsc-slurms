## Paths
Output directory for this job's own outputs (h5, stlfp.npy, stpc.npy), one folder per pid:
`cd /mnt/home/owinter/ceph/ea/cells`

LFP preprocessing outputs (lfpack / lfp-encoders — a separate tree, not written by this
job), one folder per pid:
`cd /mnt/home/owinter/ceph/ea/denoised_lfp`

Output directory for aggregate tables (written by `aggregate.py`):
`cd /mnt/home/owinter/ceph/ea/cells_aggregates` (and `cells_aggregates_f32` for the float32 archive)

Code directory on Popeye:
`cd /mnt/home/owinter/Documents/sdsc-slurms/2026-03_EA_Cells`
`cd /mnt/home/owinter/Documents/ephys-atlas`

## Compute steps

`cells.py` runs one step at a time across all PIDs via `--step {cells,stlfp,stpc}`
(default `cells`), parallelised with joblib (48 workers). Each step is idempotent —
failed PIDs write a `{pid}_{step}.error` traceback file instead of raising, so a job
can safely be re-submitted to pick up only what's missing.

| Step    | Function        | Writes                        | Depends on |
|---------|-----------------|--------------------------------|------------|
| `cells` | `cell_features` | `{pid}.h5`                    | spike sorting only |
| `stlfp` | `stlfp`         | `stlfp.npy`                    | `denoised_lfp/<pid>/lf_resampled_car_cadzow.npy` (lfpack) |
| `stpc`  | `stpc`          | `stpc.npy`                    | spike sorting only |

`{pid}.h5` holds waveforms, log-binned ACGs, burstiness/memory, and (if `--acg3d`) the 3D ACGs — see [`{pid}.h5` format](#pidh5-format) below for the full layout.

Run order: `cells` and `stpc` only need the spike sorting and can run anytime.
`stlfp` needs `denoised_lfp/<pid>/lf_resampled_car_cadzow.npy` to already exist, so run
the lfpack compression job (`../2026-06-lfpack`) first.

### `cells` step — per-cluster features (waveforms, log-ACG, 3D ACG)
```bash
sbatch cells.sbatch                        # skip PIDs that already have {pid}.h5
sbatch cells.sbatch --overwrite            # recompute every PID from scratch
sbatch cells.sbatch --acg3d                # also compute the 3D (firing-rate x time-lag) ACG, all clusters
sbatch cells.sbatch --overwrite --acg3d    # recompute everything, including 3D ACGs
```
`--acg3d` is opt-in because it's substantially more expensive than the log-binned
ACG (`acgs_log_bins`) — it's computed over all ~925k clusters, not just good units.

### `stlfp` step — spike-triggered LFP
```bash
sbatch cells.sbatch --step stlfp
```
Reads `lf_resampled_car_cadzow.npy` from the separate `denoised_lfp` tree — the
Cadzow-denoised, resampled LFP checkpoint produced by the lfpack job
(`../2026-06-lfpack/compress.py`), which is the source of truth for resampled LFP.
Writes `stlfp.npy` into `cells/<pid>/`. This step does not check for an existing
`stlfp.npy` before running, so re-submitting recomputes every PID.

### `stpc` step — spike-triggered population coupling
```bash
sbatch cells.sbatch --step stpc
```
Writes `stpc.npy` (good clusters only) atomically — computed into `stpc.npy.tmp`,
renamed into place on success — and skips a PID if `stpc.npy` already exists.
`coupling_delay`/`coupling_strength` are no longer persisted per pid; `aggregate.py`
derives them straight from the concatenated `stpc.npy` arrays (see `compute_coupling_metrics`).

### Aggregate — reduce all PIDs into one set of tables
Run once cells-step (and, if wanted, stlfp/stpc) are done for all PIDs:
```bash
python aggregate.py
```
See the docstring at the top of `aggregate.py` for the full list of output files/shapes.
`clusters.acgs_3d.npy` is only written if every `{pid}.h5` has an `acgs_3d` dataset
(i.e. the whole `cells` step was run with `--acg3d`) — otherwise it's skipped with a
warning to avoid silently misaligning rows with `clusters.table.pqt`.

## Other useful commands (progress check / rsync)

`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cells_aggregates /Users/olivier/Documents/datadisk/paper-ephys-atlas/cells_aggregates`
`rsync -av --progress -e ssh popeye:/mnt/home/owinter/ceph/ea/cells_aggregates /mnt/s0/Data/paper-ephys-atlas/cells`

Check progress of the `cells` step (PIDs done):
`find /mnt/home/owinter/ceph/ea/cells -maxdepth 2 -name '*.h5' | wc -l`

Check for errors on any step:
`ls /mnt/home/owinter/ceph/ea/cells/*_*.error 2>/dev/null`


## `{pid}.h5` format
Written by `cell_features()`. `n_clusters` = all clusters for the insertion; `n_good`
= clusters with `bitwise_fail == 0`; `nc` = channels per cluster's local waveform
neighbourhood; `ns` = samples per waveform; `total_nb_traces` = sum over clusters of
valid (in-bounds) neighbourhood channels.

Arrays (`h5py`, gzip level 4 where large):

| Dataset                    | Shape                          | dtype   | Content |
|-----------------------------|--------------------------------|---------|---------|
| `avg_waveforms`             | `(total_nb_traces, ns)`        | float32 | Mean waveform, one row per (cluster, channel) pair in its local neighbourhood; row order given by `avg_waveforms_index` |
| `avg_waveform_peak_channel` | `(n_clusters, ns)`             | float32 | Mean waveform on each cluster's own peak channel only |
| `acgs_log_bins`             | `(n_clusters, 128)`            | float32 | Log-time-binned ACG, raw spike counts (unnormalised — `aggregate.py` divides by `spike_count`) |
| `acgs_log_times`            | `(128,)`                       | float64 | Lag-time bin centers, seconds, log-spaced from 1 ms to 2 s |
| `acgs_3d`                   | `(n_clusters, 10, 201)`        | float32 | Only if `--acg3d`. 10 firing-rate deciles x 201 time-lag bins (±1000 ms at 1 ms resolution) |

DataFrames (pandas `HDFStore`, `format='fixed'`), all indexed by `cluster_id` unless noted:

| Key                     | Columns | Content |
|--------------------------|---------|---------|
| `df_clusters`            | standard `SpikeSortingLoader.merge_clusters()` output (`label`, `firing_rate`, `amp_median`, `contamination`, anatomy, etc.) + `pid` | One row per cluster |
| `avg_waveforms_index`    | `pid`, `cluster_id`, `abs_channel` | Maps each row of `avg_waveforms` to a (cluster, channel); not indexed by cluster_id |
| `avg_waveform_features`  | `peak_time_idx`, `peak_val`, `trough_time_idx`, `trough_val`, `tip_time_idx`, `tip_val`, `half_peak_*`, `recovery_time_idx`, `recovery_val`, `depolarisation_slope`, `repolarisation_slope`, `recovery_slope`, `peak_channel`, `axial_um`, `lateral_um` | From `ibldsp.waveforms.compute_spike_features()` on `avg_waveform_peak_channel` |
| `df_clusters_extended`   | `burstiness`, `memory` | Per `ephysatlas.cells.compute_burstiness_and_memory()`; NaN if a cluster has < 6 spikes |

Attrs on the root group: `pid` (str), `n_clusters`, `n_good`, `nc` (all int).

`aggregate.py` reduces all `{pid}.h5` into the tables and arrays under `cells_aggregates/`
(all-clusters `acgs_log`/`acgs_3d`/`clusters.table.pqt`, good-only `stpc`/`stlfp`) — see
its module docstring for exact output shapes.

## Output format
Two separate per-pid trees:

`cells/<pid>/` — written by this job:
    {pid}.h5     cell_features() output (waveforms, ACGs, burstiness/memory)
    stlfp.npy
    stpc.npy

`denoised_lfp/<pid>/` — written by the lfpack job (`../2026-06-lfpack`), read by `stlfp`:
    lf_resampled_car_cadzow.npy