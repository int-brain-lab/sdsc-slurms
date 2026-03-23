import os
os.environ['TQDM_DISABLE'] = '1'
from pathlib import Path
import joblib
import traceback
import time

import pandas as pd
from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader

import ephysatlas.cells

TABLES_DIR = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
OUTPUT_PATH = Path(f'/mnt/home/owinter/ceph/ea/cells')

file_insertions = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')

df_insertions = pd.read_parquet(file_insertions)
pids = list(df_insertions.loc[df_insertions['histology'] != '', 'pid'])


def cell_features(pid):
    output_path = OUTPUT_PATH.joinpath(pid)
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
    if OUTPUT_PATH.exists:
        return
    try:
        cell_features(pid)
    except Exception:
        traceback_path = OUTPUT_PATH.joinpath(f'{pid}.error')
        traceback_path.write_text(traceback.format_exc())


jobs = [joblib.delayed(cell_features_wrapper)(pid=pid) for pid in pids]

def worker_init():
    delay = (os.getpid() % 100)  # 0..99 seconds-ish, deterministic per process
    time.sleep(delay)

joblib.Parallel(
    n_jobs=48,
    backend="loky",
    initializer=worker_init,
)(jobs)
