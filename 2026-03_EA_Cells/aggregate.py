"""
Reduce step: concatenate per-insertion HDF5 and STPC files produced by cells.py
into single arrays and dataframes saved under AGG_PATH.

Outputs (1091 insertions, 925 251 clusters, 108 606 good clusters)
-------------------------------------------------------------------
File                          Shape                dtype    Size
----                          -----                -----    ----
waveforms.voltage.npy         (36 238 663, 128)    float16  8.6 GB
clusters.waveforms_peak.npy   (925 251, 128)       float16  226 MB
clusters.acgs_log.npy         (925 251, 128)       float16  226 MB  – normalised by spike_count (sp/sp)
acgs_log.times.npy            (128,)               float64  <1 MB
clusters.acgs_3d.npy          (925 251, 10, 201)   float16  3.5 GB  – only insertions run with --acg3d
acgs_3d.times.npy             (201,)               float64  <1 MB   – log-time bin centres, ms
clusters_good.stpc.npy        (108 606, 1000)      float16  207 MB
clusters_good.stlfp.npy       (108 606, 250)       float16   52 MB

clusters.table.pqt            925 251 rows, 59 cols         209 MB
  – all clusters: QC metrics, anatomy, burstiness/memory, waveform features
clusters_good.table.pqt       108 606 rows, 61 cols          32 MB
  – good clusters (bitwise_fail == 0): same + coupling_delay / coupling_strength,
    derived from clusters_good.stpc.npy (see compute_coupling_metrics below)
    (coupling columns present only when stpc() has been run for every insertion)
waveforms.table.pqt           36 238 663 rows,  3 cols       11 MB
  – pid / cluster_id / abs_channel index into waveforms.voltage.npy
"""

from pathlib import Path
import tqdm
import h5py
import numpy as np
import pandas as pd
from ephysatlas.cells import LAG, BINSIZE

CELLS_PATH = Path('/mnt/home/owinter/ceph/ea/cells')
AGG_PATH = Path('/mnt/home/owinter/ceph/ea/cells_aggregates')
AGG_PATH_F32 = Path('/mnt/home/owinter/ceph/ea/cells_aggregates_f32')  # full-resolution archive


def compute_coupling_metrics(stpc):
    """
    Coupling strength (zero-lag STPC) and coupling delay (STPC-weighted mean lag).

    Replicates the tail end of ephysatlas.cells.spike_triggered_population_coupling()
    so these two scalars can be derived straight from clusters_good.stpc.npy, instead
    of stpc() having to also persist a per-pid clusters.pqt just for these columns.

    Parameters
    ----------
    stpc : np.ndarray
        (n_good_clusters, n_time_bins) STPC traces, as saved by stpc().

    Returns
    -------
    dict of np.ndarray
        `coupling_strength` and `coupling_delay`, one value per row of `stpc`.
    """
    tscale = np.arange(stpc.shape[1]) * BINSIZE - LAG
    i0 = np.searchsorted(tscale, 0)
    coupling_strength = stpc[:, i0]
    coupling_delay = np.sum(tscale * np.abs(stpc), axis=1) / np.sum(np.abs(stpc), axis=1)
    return {'coupling_strength': coupling_strength, 'coupling_delay': coupling_delay}


h5_files = sorted(CELLS_PATH.rglob('*.h5'))

acc_avg_waveforms = []
acc_waveforms_peak = []
acc_acgs_log = []
acc_acgs_3d = []
has_acgs_3d = True  # False as soon as one insertion lacks it (ran without --acg3d)
acc_df_clusters = []
acc_df_good = []
acc_df_avg_waveforms_index = []
acc_stpc = []
acc_stlfp = []
has_stpc = True  # False as soon as one insertion lacks stpc.npy/stlfp.npy

for fil in tqdm.tqdm(h5_files, desc='reduce'):
    with h5py.File(fil, 'r') as h5:
        acc_avg_waveforms.append(h5['avg_waveforms'][:])
        acc_waveforms_peak.append(h5['avg_waveform_peak_channel'][:])
        acc_acgs_log.append(h5['acgs_log_bins'][:])
        acgs_log_times = h5['acgs_log_times'][:]  # identical across insertions; keep last
        if 'acgs_3d' in h5:
            acc_acgs_3d.append(h5['acgs_3d'][:])
            acgs_3d_times = h5['acgs_3d_times'][:]  # identical across insertions; keep last
        else:
            has_acgs_3d = False
    df_cl = pd.read_hdf(fil, key='df_clusters')
    df_cl = df_cl.join(pd.read_hdf(fil, key='df_clusters_extended'))   # burstiness / memory
    wf_features = pd.read_hdf(fil, key='avg_waveform_features')
    wf_features = wf_features.drop(columns=wf_features.columns.intersection(df_cl.columns))
    df_cl = df_cl.join(wf_features)  # waveform shape features
    acc_df_clusters.append(df_cl)
    df_good = df_cl.loc[df_cl['bitwise_fail'] == 0].copy()
    acc_df_good.append(df_good)
    acc_df_avg_waveforms_index.append(pd.read_hdf(fil, key='avg_waveforms_index'))
    # stpc / stlfp are saved only for good clusters by cells.py; coupling_delay /
    # coupling_strength are derived from stpc below, not read from a per-pid file
    stpc_file = fil.parent.joinpath('stpc.npy')
    stlfp_file = fil.parent.joinpath('stlfp.npy')
    if stpc_file.exists() and stlfp_file.exists():
        acc_stpc.append(np.load(stpc_file))
        acc_stlfp.append(np.load(stlfp_file))
    else:
        has_stpc = False

avg_waveforms = np.concatenate(acc_avg_waveforms, axis=0)
waveforms_peak = np.concatenate(acc_waveforms_peak, axis=0)
acgs_log = np.concatenate(acc_acgs_log, axis=0)
# aligned with clusters.table.pqt (all units); only present if every insertion ran with --acg3d
acgs_3d = np.concatenate(acc_acgs_3d, axis=0) if has_acgs_3d and acc_acgs_3d else None
df_clusters = pd.concat(acc_df_clusters)
df_clusters_good = pd.concat(acc_df_good)
df_avg_waveforms_index = pd.concat(acc_df_avg_waveforms_index, ignore_index=True)
# aligned with clusters_good.table.pqt (row order matches: both built per-pid, in the
# same bitwise_fail == 0 order); only present if every insertion ran the stpc step
stpc = np.concatenate(acc_stpc, axis=0) if has_stpc and acc_stpc else None
stlfp = np.concatenate(acc_stlfp, axis=0) if has_stpc and acc_stlfp else None
if stpc is not None:
    df_clusters_good = df_clusters_good.assign(**compute_coupling_metrics(stpc))

print(f'n_insertions            {len(h5_files)}')
print(f'waveforms.voltage       {avg_waveforms.shape}')
print(f'clusters.waveforms_peak {waveforms_peak.shape}')
print(f'clusters.acgs_log       {acgs_log.shape}')
print(f'clusters.acgs_3d        {acgs_3d.shape if acgs_3d is not None else "skipped (missing in some insertions)"}')
print(f'clusters.table          {df_clusters.shape}')
print(f'clusters_good.table     {df_clusters_good.shape}')
print(f'waveforms.table         {df_avg_waveforms_index.shape}')
print(f'clusters_good.stpc      {stpc.shape if stpc is not None else "skipped (missing in some insertions)"}')
print(f'clusters_good.stlfp     {stlfp.shape if stlfp is not None else "skipped (missing in some insertions)"}')

acgs_log_norm = acgs_log / df_clusters['spike_count'].values[:, np.newaxis]

AGG_PATH.mkdir(parents=True, exist_ok=True)
np.save(AGG_PATH.joinpath('waveforms.voltage.npy'), avg_waveforms.astype(np.float16))
np.save(AGG_PATH.joinpath('clusters.waveforms_peak.npy'), waveforms_peak.astype(np.float16))
np.save(AGG_PATH.joinpath('clusters.acgs_log.npy'), acgs_log_norm.astype(np.float16))
np.save(AGG_PATH.joinpath('acgs_log.times.npy'), acgs_log_times)
if acgs_3d is not None:
    np.save(AGG_PATH.joinpath('clusters.acgs_3d.npy'), acgs_3d.astype(np.float16))
    np.save(AGG_PATH.joinpath('acgs_3d.times.npy'), acgs_3d_times)
df_clusters.to_parquet(AGG_PATH.joinpath('clusters.table.pqt'))
df_clusters_good.to_parquet(AGG_PATH.joinpath('clusters_good.table.pqt'))
df_avg_waveforms_index.to_parquet(AGG_PATH.joinpath('waveforms.table.pqt'))
if stpc is not None:
    np.save(AGG_PATH.joinpath('clusters_good.stpc.npy'), stpc.astype(np.float16))
    np.save(AGG_PATH.joinpath('clusters_good.stlfp.npy'), stlfp.astype(np.float16))

AGG_PATH_F32.mkdir(parents=True, exist_ok=True)
np.save(AGG_PATH_F32.joinpath('waveforms.voltage.npy'), avg_waveforms.astype(np.float32))
np.save(AGG_PATH_F32.joinpath('clusters.waveforms_peak.npy'), waveforms_peak.astype(np.float32))
np.save(AGG_PATH_F32.joinpath('clusters.acgs_log.npy'), acgs_log_norm.astype(np.float32))
if acgs_3d is not None:
    np.save(AGG_PATH_F32.joinpath('clusters.acgs_3d.npy'), acgs_3d.astype(np.float32))
if stpc is not None:
    np.save(AGG_PATH_F32.joinpath('clusters_good.stpc.npy'), stpc.astype(np.float32))
    np.save(AGG_PATH_F32.joinpath('clusters_good.stlfp.npy'), stlfp.astype(np.float32))
