import argparse
import os
os.environ['TQDM_DISABLE'] = '1'
from pathlib import Path
import joblib
import traceback
import time
import ibldsp.waveforms

import h5py
import numpy as np
import pandas as pd
from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader

from slidingRP.metrics import slidingRP_all

import ephysatlas.cells
from ephysatlas.cells import compute_log_acg, compute_burstiness_and_memory, compute_3d_acgs

EXCLUDES = [
    '1122230f-42b8-45f9-ad7e-00c27ae087c8',  # dartsort error
    '1dd218c9-ac97-4d91-80d0-a8a660bf7395',  # dartsort broadcast
    '316a733a-5358-4d1d-9f7f-179ba3c90adf',  # dartsort error
    '71a92c54-69f0-488b-ae2a-cb6c1524233c',  # dartsort error
    '80494687-eb74-43c6-801c-e99fd6621d51',  # dartsort broadcast
    'fb76fd5c-0b91-41f2-9b94-0f64b62396cb',  # dartsort broadcast
    'ce16c71a-f0a6-48b7-bc2f-430ff94df5de',  # spike sorting stuck Elbocal
    'f362c84f-8d9a-4d5b-8439-055ae936fdff',  # spike sorting stuck Elbocal
]

TABLES_DIR = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
OUTPUT_PATH = Path('/mnt/home/owinter/ceph/ea/cells')          # cell features: h5, stlfp.npy, stpc.npy
LFP_PATH = Path('/mnt/home/owinter/ceph/ea/denoised_lfp')       # LFP preprocessing outputs (lfpack)

# ── ACG parameters ────────────────────────────────────────────────────────────
BIN_SIZE = 0.2e-3   # s base resolution
WIN_SIZE = 2.0      # s one-sided window
N_LOG_BINS = 128
LOG_TRIM = 1e-3     # s start of log axis (refractory period cutoff)

# 3D ACG parameters/computation now live in ephysatlas.cells (compute_3d_acgs,
# ACG3D_* constants) — shared with any non-batch recompute (e.g. inference on a
# new dataset), not duplicated here.

# ── Sliding RP v2 parameters ───────────────────────────────────────────────────
# Forwarded as-is to compute_sliding_rp_v2 / patch_sliding_rp_v2 (cells and
# patch_slidingrp steps). Edit and re-run --step patch_slidingrp to re-tune
# without recomputing waveforms/ACGs.
SLIDING_RP_PARAMS = dict(conf_thresh=70, cont_thresh=10, rp_reject=0.0005, force_pass=True)

# Hard refractory period below which a cluster is flagged as contaminated: a
# cluster with any inter-spike interval shorter than this cannot be a single unit.
RP_CONTAM_THRESH = 1.5e-3  # s

file_insertions = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')

df_insertions = pd.read_parquet(file_insertions)
pids = list(df_insertions.loc[df_insertions['histology'] != '', 'pid'])


def compute_sliding_rp_v2(spikes, df_clusters, fs, rec_dur, conf_thresh=90,
                          cont_thresh=10, rp_reject=0.0005, force_pass=True):
    """
    New sliding refractory-period QC metric (SteinmetzLab/slidingRefractory v2),
    all clusters.

    Columns are prefixed `slidingRP2_` so they cannot collide with the legacy
    `slidingRP_viol` / `slidingRP_viol_forced` already in `df_clusters` (computed
    by the previous, buggy implementation as part of spike sorting).

    Parameters
    ----------
    spikes : dict
        IBL spikes dict with 'times' (s) and 'clusters' arrays for the insertion.
    df_clusters : pd.DataFrame
        Indexed by cluster_id; used only to fix the row order/index of the result,
        so clusters with no spikes at all still get a (NaN) row.
    fs : float
        AP-band sampling rate, in Hz.
    rec_dur : float
        Recording duration, in seconds (e.g. sr.ns / sr.fs).
    conf_thresh, cont_thresh, rp_reject, force_pass
        Forwarded to `slidingRP.metrics.slidingRP_all` — see its docstring.
        Change these here (or at the call site) to re-tune the metric; use
        `patch_sliding_rp_v2()` to re-apply to already-computed {pid}.h5 files
        without recomputing waveforms/ACGs.

    Returns
    -------
    pd.DataFrame
        Indexed like `df_clusters`, columns: slidingRP2_max_confidence,
        slidingRP2_min_contamination, slidingRP2_rp_min_val,
        slidingRP2_n_spikes_below2, slidingRP2_firing_rate, slidingRP2_viol,
        slidingRP2_viol_forced, is_contaminated.
    """
    rp = slidingRP_all(
        spikes['times'], spikes['clusters'],
        params={'recDur': rec_dur, 'sampleRate': fs, 'forcePass': force_pass},
        conf_thresh=conf_thresh, cont_thresh=cont_thresh, rp_reject=rp_reject,
        n_jobs=1,  # insertions are already parallelised across workers below
    )
    df_rp = pd.DataFrame({
        'slidingRP2_max_confidence': rp['max_confidence'],
        'slidingRP2_min_contamination': rp['min_contamination'],
        'slidingRP2_rp_min_val': rp['rp_min_val'],
        'slidingRP2_n_spikes_below2': rp['n_spikes_below2'],
        'slidingRP2_firing_rate': rp['firing_rate'],
        'slidingRP2_viol': rp['value'],
        'slidingRP2_viol_forced': rp['value_forced'],
    }, index=rp['cidx'])

    # Contamination flag: True when the shortest inter-spike interval of a cluster
    # falls within the hard refractory period (RP_CONTAM_THRESH). spikes['times']
    # is globally time-sorted, so each per-cluster subset stays time-ordered and a
    # single groupby-diff yields the min ISI without re-sorting.
    times = pd.Series(spikes['times'], index=spikes['clusters'])
    min_isi = times.groupby(level=0).diff().groupby(level=0).min()
    df_rp['is_contaminated'] = (min_isi < RP_CONTAM_THRESH).reindex(df_rp.index, fill_value=False)

    return df_rp.reindex(df_clusters.index)


def patch_sliding_rp_v2(pid, **rp_kwargs):
    """
    Recompute only the slidingRP2_* QC columns for one insertion and patch them
    into the existing {pid}.h5 — without recomputing waveforms or ACGs.

    Use this after changing slidingRP parameters (pass them as `rp_kwargs`, e.g.
    `conf_thresh=95`) or fixing a bug in `compute_sliding_rp_v2`, instead of
    re-running the whole `cells` step. Still needs to reload `spikes` (not cached
    in the h5) and `sr.fs`/`sr.ns` (cheap metadata), but skips everything else.

    Every other array/dataframe is copied across unchanged via h5py's raw
    `copy()` (preserves compression/attrs exactly) into a fresh tmp file, which
    is then renamed over the original — same atomic discipline as
    cell_features(), this time protecting the expensive, already-computed parts
    of the file from a crash mid-patch.

    Parameters
    ----------
    pid : str
        Probe insertion UUID; {pid}.h5 must already exist.
    **rp_kwargs
        Forwarded to `compute_sliding_rp_v2` (conf_thresh, cont_thresh,
        rp_reject, force_pass).
    """
    outfile = OUTPUT_PATH.joinpath(pid, f'{pid}.h5')
    outfile_tmp = outfile.with_suffix('.h5.tmp')

    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, _ = ssl.load_spike_sorting(dataset_types=['spikes.times', 'spikes.clusters'])
    sr = ssl.raw_electrophysiology(band='ap', stream=True)

    df_clusters = pd.read_hdf(outfile, key='df_clusters')
    df_clusters_extended = pd.read_hdf(outfile, key='df_clusters_extended')

    # Safety: the patch aligns freshly-loaded spikes to the *stored* df_clusters by
    # integer cluster id (0..N). A re-run spike sorting would reuse the same ids for
    # different units and silently misalign the patched columns; clusters.uuids are
    # stable per unit, so refuse to patch on any mismatch. The wrapper turns this into
    # a {pid}_patch_slidingrp.error and leaves the existing {pid}.h5 untouched.
    fresh_uuids = np.asarray(clusters['uuids']).ravel()
    stored_uuids = df_clusters['uuids'].to_numpy()
    if not np.array_equal(fresh_uuids, stored_uuids):
        raise ValueError(
            f'{pid}: clusters.uuids from ONE no longer match stored {pid}.h5 '
            f'({fresh_uuids.size} vs {stored_uuids.size} clusters); '
            f'spike sorting changed — refusing to patch.'
        )
    df_clusters_extended = df_clusters_extended.drop(
        columns=[c for c in df_clusters_extended.columns
                 if c.startswith('slidingRP2_') or c == 'is_contaminated']
    )
    df_rp2 = compute_sliding_rp_v2(spikes, df_clusters, sr.fs, rec_dur=sr.ns / sr.fs, **rp_kwargs)
    df_clusters_extended = df_clusters_extended.join(df_rp2)

    with h5py.File(outfile, 'r') as h5_src, h5py.File(outfile_tmp, 'w') as h5_dst:
        for key in h5_src.keys():
            if key != 'df_clusters_extended':
                h5_src.copy(key, h5_dst)
        h5_dst.attrs.update(h5_src.attrs)
    df_clusters_extended.to_hdf(outfile_tmp, key='df_clusters_extended', mode='a', format='fixed')
    outfile_tmp.rename(outfile)


def patch_acg3d(pid):
    """
    Compute the 3D (firing-rate decile x log-time-lag) ACG for one insertion and
    patch it into the existing {pid}.h5 as `acgs_3d`/`acgs_3d_times` — without
    recomputing waveforms, log-ACGs, or the slidingRP2_* QC columns.

    Use this to add `acgs_3d` to insertions that were run without `--acg3d`,
    instead of `--overwrite --acg3d` (which redoes everything). Always
    recomputes and overwrites `acgs_3d`/`acgs_3d_times` if already present.

    Every other array/dataframe is copied across unchanged via h5py's raw
    `copy()` (preserves compression/attrs exactly) into a fresh tmp file, which
    is then renamed over the original — same atomic discipline as
    cell_features(), protecting the already-computed parts of the file from a
    crash mid-patch.

    Parameters
    ----------
    pid : str
        Probe insertion UUID; {pid}.h5 must already exist.
    """
    outfile = OUTPUT_PATH.joinpath(pid, f'{pid}.h5')
    outfile_tmp = outfile.with_suffix('.h5.tmp')

    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, _ = ssl.load_spike_sorting(dataset_types=['spikes.times', 'spikes.clusters'])
    sr = ssl.raw_electrophysiology(band='ap', stream=True)

    df_clusters = pd.read_hdf(outfile, key='df_clusters')

    # Safety: acgs_3d rows are aligned to the *stored* df_clusters by integer cluster
    # id, which is only meaningful if the underlying units are unchanged. clusters.uuids
    # are stable per unit, so refuse to patch on any mismatch (a re-run spike sorting).
    # The wrapper turns this into a {pid}_patch_acg3d.error and leaves {pid}.h5 untouched.
    fresh_uuids = np.asarray(clusters['uuids']).ravel()
    stored_uuids = df_clusters['uuids'].to_numpy()
    if not np.array_equal(fresh_uuids, stored_uuids):
        raise ValueError(
            f'{pid}: clusters.uuids from ONE no longer match stored {pid}.h5 '
            f'({fresh_uuids.size} vs {stored_uuids.size} clusters); '
            f'spike sorting changed — refusing to patch.'
        )

    acgs_3d, acgs_3d_times = compute_3d_acgs(spikes['times'], spikes['clusters'], df_clusters.index.values, sr.fs)

    with h5py.File(outfile, 'r') as h5_src, h5py.File(outfile_tmp, 'w') as h5_dst:
        for key in h5_src.keys():
            if key not in ('acgs_3d', 'acgs_3d_times'):
                h5_src.copy(key, h5_dst)
        h5_dst.attrs.update(h5_src.attrs)
        h5_dst.create_dataset('acgs_3d', data=acgs_3d, compression='gzip', compression_opts=4)
        h5_dst.create_dataset('acgs_3d_times', data=acgs_3d_times.astype(np.float64))
    outfile_tmp.rename(outfile)


def cell_features(pid, overwrite=False, compute_3dacg=False, rp_kwargs=None):
    """
    Extract per-cluster features for one insertion and save to a single HDF5 file.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    overwrite : bool
        If True, delete any existing output file and recompute.
    compute_3dacg : bool
        If True, also compute the 3D (firing-rate decile x log-time-lag) ACG for
        all clusters and write it as `acgs_3d`/`acgs_3d_times`. Off by default:
        much more expensive than `acgs_log_bins`.
    rp_kwargs : dict, optional
        Forwarded to `compute_sliding_rp_v2` (conf_thresh, cont_thresh, rp_reject,
        force_pass). Defaults there apply if not given.

    Outputs written to OUTPUT_PATH / pid / {pid}.h5:
      Arrays (h5py, gzip-compressed where large):
        avg_waveforms            (total_nb_traces, ns)     valid neighbourhood traces, flat
        avg_waveform_peak_channel (n_clusters, ns)
        acgs_log_bins            (n_clusters, N_LOG_BINS)
        acgs_log_times           (N_LOG_BINS,)
        acgs_3d                  (n_clusters, 10, 201)      only if compute_3dacg=True
        acgs_3d_times            (201,)                     only if compute_3dacg=True
      DataFrames (pandas HDFStore, appended):
        df_clusters              cluster table after ssl merge
        avg_waveforms_index      pid / cluster_id / abs_channel for each flat trace row
        avg_waveform_features    waveform shape features
        df_clusters_extended     burstiness, memory, and slidingRP2_* QC columns
                                  per cluster
    """
    outdir = OUTPUT_PATH.joinpath(pid)
    outfile = outdir.joinpath(f'{pid}.h5')
    if overwrite:
        outfile.unlink(missing_ok=True)
        outdir.joinpath(f'{pid}.h5.tmp').unlink(missing_ok=True)
    elif outfile.exists():
        return

    outdir.mkdir(parents=True, exist_ok=True)
    outfile_tmp = outdir.joinpath(f'{pid}.h5.tmp')
    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting()
    df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))
    df_clusters['pid'] = pid
    sr = ssl.raw_electrophysiology(band='ap', stream=True)
    good_ids = df_clusters.index.values[df_clusters['bitwise_fail'] == 0]
    n_good = good_ids.size

    # Waveforms
    avg_waveforms = ssl.load_spike_sorting_object('waveforms', attribute=['templates'])['templates']
    wxy, winds = ibldsp.waveforms.get_waveforms_coordinates(
        trace_indices=clusters['channels'],
        xy=np.c_[sr.geometry['x'], sr.geometry['y']],
        return_indices=True,
    )
    df_wf_features = ibldsp.waveforms.compute_spike_features(
        avg_waveforms.transpose(0, 2, 1), fs=sr.fs
    )
    df_wf_features.index = df_clusters.index
    peak_j = df_wf_features['peak_trace_idx'].values.astype(int)
    peak_i = np.arange(len(df_wf_features))
    df_wf_features['peak_channel'] = winds[peak_i, peak_j].astype(np.int16)
    df_wf_features['axial_um'] = wxy[peak_i, peak_j, 1].astype(np.float32)
    df_wf_features['lateral_um'] = wxy[peak_i, peak_j, 0].astype(np.float32)
    df_wf_features = df_wf_features.drop(columns='peak_trace_idx')

    n_clusters, nc, ns = avg_waveforms.shape
    n_channels = sr.geometry['x'].size
    valid_mask = winds.reshape(-1) < n_channels
    avg_waveforms_flat = avg_waveforms.reshape(-1, ns)[valid_mask].astype(np.float32)
    avg_waveform_peak_channel = avg_waveforms[peak_i, peak_j, :].astype(np.float32)
    df_avg_waveforms_index = pd.DataFrame({
        'pid': pid,
        'cluster_id': np.repeat(df_clusters.index.values, nc)[valid_mask],
        'abs_channel': winds.reshape(-1)[valid_mask].astype(np.int16),
    })

    # ACGs (all clusters, log-binned)
    acgs_log_bins, acgs_log_times = compute_log_acg(
        spikes['times'], sr.fs, spike_clusters=spikes['clusters'],
        bin_size=BIN_SIZE, win_size=WIN_SIZE, n_log_bins=N_LOG_BINS, log_trim=LOG_TRIM,
    )

    # 3D ACGs (all clusters, opt-in: expensive)
    if compute_3dacg:
        acgs_3d, acgs_3d_times = compute_3d_acgs(spikes['times'], spikes['clusters'], df_clusters.index.values, sr.fs)

    # Burstiness and memory (all clusters)
    bm = np.array(
        [compute_burstiness_and_memory(spikes['times'][spikes['clusters'] == cid])
         for cid in df_clusters.index],
        dtype=np.float32,
    )
    df_clusters_extended = pd.DataFrame(bm, index=df_clusters.index, columns=['burstiness', 'memory'])

    # Sliding RP v2 QC metric (all clusters)
    df_rp2 = compute_sliding_rp_v2(spikes, df_clusters, sr.fs, rec_dur=sr.ns / sr.fs, **(rp_kwargs or {}))
    df_clusters_extended = df_clusters_extended.join(df_rp2)

    # Write to a tmp file first; rename to final path only on full success.
    # This ensures a killed/crashed job never leaves a partial file that the
    # skip-if-exists guard would silently accept on the next run.
    with h5py.File(outfile_tmp, 'w') as h5:
        h5.create_dataset('avg_waveforms', data=avg_waveforms_flat,
                          compression='gzip', compression_opts=4)
        h5.create_dataset('avg_waveform_peak_channel', data=avg_waveform_peak_channel)
        h5.create_dataset('acgs_log_bins', data=acgs_log_bins.astype(np.float32),
                          compression='gzip', compression_opts=4)
        h5.create_dataset('acgs_log_times', data=acgs_log_times.astype(np.float64))
        if compute_3dacg:
            h5.create_dataset('acgs_3d', data=acgs_3d,
                              compression='gzip', compression_opts=4)
            h5.create_dataset('acgs_3d_times', data=acgs_3d_times.astype(np.float64))
        h5.attrs['pid'] = pid
        h5.attrs['n_clusters'] = n_clusters
        h5.attrs['n_good'] = n_good
        h5.attrs['nc'] = nc
    df_clusters.to_hdf(outfile_tmp, key='df_clusters', mode='a', format='fixed')
    df_avg_waveforms_index.to_hdf(outfile_tmp, key='avg_waveforms_index', mode='a', format='fixed')
    df_wf_features.to_hdf(outfile_tmp, key='avg_waveform_features', mode='a', format='fixed')
    df_clusters_extended.to_hdf(outfile_tmp, key='df_clusters_extended', mode='a', format='fixed')
    outfile_tmp.rename(outfile)


def stlfp(pid):
    """
    Spike-triggered LFP for one insertion.

    Reads `lf_resampled_car_cadzow.npy` from LFP_PATH — the Cadzow-denoised, resampled
    LFP checkpoint produced by the separate lfpack job (../2026-06-lfpack/compress.py),
    the source of truth for resampled LFP. Requires that job to have run first.
    """
    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

    file_rsamp_lfp = LFP_PATH.joinpath(pid, 'lf_resampled_car_cadzow.npy')
    file_stlfp = OUTPUT_PATH.joinpath(pid, 'stlfp.npy')
    file_stlfp.parent.mkdir(parents=True, exist_ok=True)

    ephysatlas.cells.spike_triggered_lfp(
        file_rsamp_lfp,
        spikes,
        df_clusters,
        event_window=(-0.5, 0.5),
        fs_ap=30_000,
        fs=2500 // 10,
        file_stlfp=file_stlfp
    )


def stpc(pid):
    """
    Spike-triggered population coupling for one insertion.

    Writes only `stpc.npy` (good clusters), atomically: computed into a `.tmp` file
    via `file_stpc`, then renamed into place, so a killed job never leaves a partial
    `stpc.npy` that the skip-guard below would mistake for a completed run.
    """
    output_path = OUTPUT_PATH.joinpath(pid)
    outfile = output_path.joinpath('stpc.npy')
    if outfile.exists():
        return
    output_path.mkdir(parents=True, exist_ok=True)
    outfile_tmp = output_path.joinpath('stpc.npy.tmp')
    outfile_tmp.unlink(missing_ok=True)  # drop any partial file left by a previous crash

    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting()
    df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

    ephysatlas.cells.spike_triggered_population_coupling(
        spikes,
        df_clusters,
        file_stpc=outfile_tmp,
    )
    outfile_tmp.rename(outfile)


def cell_features_wrapper(pid, overwrite=False, compute_3dacg=False, rp_kwargs=None):
    try:
        cell_features(pid, overwrite=overwrite, compute_3dacg=compute_3dacg, rp_kwargs=rp_kwargs)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_cell_features.error')
        traceback_path.write_text(traceback.format_exc())


def stlfp_wrapper(pid):
    try:
        stlfp(pid)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_stlfp.error')
        traceback_path.write_text(traceback.format_exc())


def stpc_wrapper(pid):
    try:
        stpc(pid)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_stpc.error')
        traceback_path.write_text(traceback.format_exc())


def patch_sliding_rp_v2_wrapper(pid, rp_kwargs=None):
    try:
        patch_sliding_rp_v2(pid, **(rp_kwargs or {}))
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_patch_slidingrp.error')
        traceback_path.write_text(traceback.format_exc())


def patch_acg3d_wrapper(pid):
    try:
        patch_acg3d(pid)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_patch_acg3d.error')
        traceback_path.write_text(traceback.format_exc())


def worker_init():
    delay = (os.getpid() % 100)  # 0..99 s stagger to avoid thundering-herd on ONE auth
    time.sleep(delay)


if __name__ == '__main__':
    # Guarded so cells.py can be imported (e.g. for a single-PID smoke test) without
    # triggering the full batch dispatch below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['cells', 'stlfp', 'stpc', 'patch_slidingrp', 'patch_acg3d'],
                        default='cells', help='which per-insertion step to run (see README)')
    parser.add_argument('--overwrite', action='store_true', help='[cells step] recompute even if HDF5 already exists')
    parser.add_argument('--acg3d', action='store_true', help='[cells step] also compute 3D ACGs for all clusters')
    args = parser.parse_args()

    if args.step == 'cells':
        jobs = [
            joblib.delayed(cell_features_wrapper)(
                pid=pid, overwrite=args.overwrite, compute_3dacg=args.acg3d, rp_kwargs=SLIDING_RP_PARAMS)
            for pid in pids if pid not in EXCLUDES
        ]
    elif args.step == 'stlfp':
        jobs = [joblib.delayed(stlfp_wrapper)(pid=pid) for pid in pids if pid not in EXCLUDES]
    elif args.step == 'stpc':
        jobs = [joblib.delayed(stpc_wrapper)(pid=pid) for pid in pids if pid not in EXCLUDES]
    elif args.step == 'patch_slidingrp':
        jobs = [
            joblib.delayed(patch_sliding_rp_v2_wrapper)(pid=pid, rp_kwargs=SLIDING_RP_PARAMS)
            for pid in pids if pid not in EXCLUDES and OUTPUT_PATH.joinpath(pid, f'{pid}.h5').exists()
        ]
    else:
        jobs = [
            joblib.delayed(patch_acg3d_wrapper)(pid=pid)
            for pid in pids if pid not in EXCLUDES and OUTPUT_PATH.joinpath(pid, f'{pid}.h5').exists()
        ]

    joblib.Parallel(
        n_jobs=48,
        backend="loky",
        initializer=worker_init,
    )(jobs)