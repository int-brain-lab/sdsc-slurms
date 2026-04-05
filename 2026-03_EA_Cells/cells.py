import os
os.environ['TQDM_DISABLE'] = '1'
from pathlib import Path
import joblib
import traceback
import time
import ibldsp.voltage

import pandas as pd
from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader

import ephysatlas.cells

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

file_insertions = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')

df_insertions = pd.read_parquet(file_insertions)
pids = list(df_insertions.loc[df_insertions['histology'] != '', 'pid'])


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
        # stpc(pid)
        # resample_lfp(pid)
        stlfp(pid)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}_stpc.error')
        traceback_path.write_text(traceback.format_exc())

jobs = [joblib.delayed(cell_features_wrapper)(pid=pid) for pid in pids if pid not in EXCLUDES]

def worker_init():
    delay = (os.getpid() % 100)  # 0..99 seconds-ish, deterministic per process
    time.sleep(delay)

joblib.Parallel(
    n_jobs=48,
    backend="loky",
    initializer=worker_init,
)(jobs)
