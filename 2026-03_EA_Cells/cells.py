import os
os.environ['TQDM_DISABLE'] = '1'
from pathlib import Path
import joblib
import traceback
import time
import ibldsp.voltage
import ibldsp.waveforms

import h5py
import numpy as np
import pandas as pd
from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader

import ephysatlas.cells
from ephysatlas.cells import compute_log_acg, compute_burstiness_and_memory

EXCLUDES = [
    '1122230f-42b8-45f9-ad7e-00c27ae087c8',  # dartsort error
    '1dd218c9-ac97-4d91-80d0-a8a660bf7395',  # dartsort broadcast
    '316a733a-5358-4d1d-9f7f-179ba3c90adf',  # dartsort error
    '71a92c54-69f0-488b-ae2a-cb6c1524233c',  # dartsort error
    '80494687-eb74-43c6-801c-e99fd6621d51',  # dartsort broadcast
    'fb76fd5c-0b91-41f2-9b94-0f64b62396cb',  # dartsort broadcast
    'ce16c71a-f0a6-48b7-bc2f-430ff94df5de',  # spike sorting stuck Elbocal
]

TABLES_DIR = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
OUTPUT_PATH = Path(f'/mnt/home/owinter/ceph/ea/cells')
Q = 10

# ── ACG parameters ────────────────────────────────────────────────────────────
BIN_SIZE = 0.2e-3   # s base resolution
WIN_SIZE = 2.0      # s one-sided window
N_LOG_BINS = 128
LOG_TRIM = 1e-3     # s start of log axis (refractory period cutoff)

file_insertions = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')

df_insertions = pd.read_parquet(file_insertions)
pids = list(df_insertions.loc[df_insertions['histology'] != '', 'pid'])


def cell_features(pid):
    """
    Extract per-cluster features for one insertion and save to a single HDF5 file.

    Outputs written to OUTPUT_PATH / pid / {pid}.h5:
      Arrays (h5py, gzip-compressed where large):
        avg_waveforms            (total_nb_traces, ns)     valid neighbourhood traces, flat
        avg_waveform_peak_channel (n_clusters, ns)
        acgs_log_bins            (n_clusters, N_LOG_BINS)
        acgs_log_times           (N_LOG_BINS,)
      DataFrames (pandas HDFStore, appended):
        df_clusters              cluster table after ssl merge
        avg_waveforms_index      pid / cluster_id / abs_channel for each flat trace row
        avg_waveform_features    waveform shape features
        df_clusters_extended     burstiness and memory per cluster
    """
    outdir = OUTPUT_PATH.joinpath(pid)
    outfile = outdir.joinpath(f'{pid}.h5')
    if outfile.exists():
        return

    outdir.mkdir(parents=True, exist_ok=True)
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

    # Burstiness and memory (all clusters)
    bm = np.array(
        [compute_burstiness_and_memory(spikes['times'][spikes['clusters'] == cid])
         for cid in df_clusters.index],
        dtype=np.float32,
    )
    df_clusters_extended = pd.DataFrame(bm, index=df_clusters.index, columns=['burstiness', 'memory'])

    # Save to HDF5
    with h5py.File(outfile, 'w') as h5:
        h5.create_dataset('avg_waveforms', data=avg_waveforms_flat,
                          compression='gzip', compression_opts=4)
        h5.create_dataset('avg_waveform_peak_channel', data=avg_waveform_peak_channel)
        h5.create_dataset('acgs_log_bins', data=acgs_log_bins.astype(np.float32),
                          compression='gzip', compression_opts=4)
        h5.create_dataset('acgs_log_times', data=acgs_log_times.astype(np.float64))
        h5.attrs['pid'] = pid
        h5.attrs['n_clusters'] = n_clusters
        h5.attrs['n_good'] = n_good
        h5.attrs['nc'] = nc
    df_clusters.to_hdf(outfile, key='df_clusters', mode='a', format='fixed')
    df_avg_waveforms_index.to_hdf(outfile, key='avg_waveforms_index', mode='a', format='fixed')
    df_wf_features.to_hdf(outfile, key='avg_waveform_features', mode='a', format='fixed')
    df_clusters_extended.to_hdf(outfile, key='df_clusters_extended', mode='a', format='fixed')


def resample_lfp(pid):
    file_rsamp_lfp = OUTPUT_PATH.joinpath(pid, 'lf_resampled.npy')
    if not file_rsamp_lfp.exists():
        one = ONE()
        ssl = SpikeSortingLoader(one=one, pid=pid)
        sr = ssl.raw_electrophysiology(band='lf', stream=False)
        channel_labels = ibldsp.voltage.detect_bad_channels_cbin(sr, display=False)
        ibldsp.voltage.resample_denoise_lfp_cbin(
            lf_file=sr, output=file_rsamp_lfp, channel_labels=channel_labels, q=Q)


def stlfp(pid):
    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

    file_rsamp_lfp = OUTPUT_PATH.joinpath(pid, 'lf_resampled.npy')
    file_stlfp = OUTPUT_PATH.joinpath(pid, 'stlfp.npy')

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
    output_path = OUTPUT_PATH.joinpath(pid)
    if output_path.joinpath('stpc.png').exists():
        return
    output_path.mkdir(parents=True, exist_ok=True)
    one = ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting()
    df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

    df_clusters, stpc, tscale, coupling_strength, taper, coupling_delay, firing_rates = (
        ephysatlas.cells.spike_triggered_population_coupling(
            spikes,
            df_clusters,
            file_stpc=output_path.joinpath('stpc.npy'),
        )
    )
    df_clusters.to_parquet(output_path.joinpath('clusters.pqt'))
    ephysatlas.cells.display_stpc(
        df_clusters,
        stpc,
        tscale,
        coupling_strength,
        coupling_delay,
        firing_rates,
        save_file=output_path.joinpath('stpc.png'),
        label=pid,
    )


def cell_features_wrapper(pid):
    try:
        cell_features(pid)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_cell_features.error')
        traceback_path.write_text(traceback.format_exc())


jobs = [joblib.delayed(cell_features_wrapper)(pid=pid) for pid in pids if pid not in EXCLUDES]


def worker_init():
    delay = (os.getpid() % 100)  # 0..99 s stagger to avoid thundering-herd on ONE auth
    time.sleep(delay)


joblib.Parallel(
    n_jobs=48,
    backend="loky",
    initializer=worker_init,
)(jobs)