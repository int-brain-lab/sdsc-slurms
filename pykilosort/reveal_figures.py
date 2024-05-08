# module load python/3.10.10
# source /mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/activate
# sbatch ~/Documents/PYTHON/sdsc-slurms/pykilosort/disbatch_reveal_figures.sbatch
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import one.alf.io as alfio
from one.api import ONE
from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader, CLUSTERS_ATTRIBUTES, SPIKES_ATTRIBUTES

from pykilosort import compare
from brainwidemap import bwm_query
from brainbox.io.one import EphysSessionLoader, SpikeSortingLoader

# iblqm / matplotlib_venn

# this is my way of loading the data on parede for the benchmark runs, but this will change once we have the data registered
sns.set_theme('paper', 'white')
one = ONE()
bwm_df = bwm_query(freeze='2023_12_bwm_release', one=one, return_details=True)

version = '1.7.0'
OVERWRITE = False
quarantine_paths = [
    Path('/mnt/sdceph/users/ibl/data/quarantine/tasks/SpikeSorting'),
    Path('/mnt/sdceph/users/ibl/data/quarantine/tasks_olivier/SpikeSorting'),
    Path('/mnt/sdceph/users/ibl/data/quarantine/tasks_owinter/SpikeSorting'),
]
path_reveal_figures = Path(f"/mnt/home/owinter/ceph/2024_rerun/reveal_figures")
path_reveal_figures.joinpath('parquet').mkdir(exist_ok=True, parents=True)

# %%
IMIN = 0
for i, rec in bwm_df.iterrows():
    if i < IMIN:
        continue
    file_pqt = path_reveal_figures.joinpath('parquet', rec.pid + '_clusters.pqt')
    if file_pqt.exists() and not OVERWRITE:
        print(i, rec.pid, 'already exists: SKIP !')
        continue
    ssl = SpikeSortingLoader(one=one, pid=rec.pid)
    for quarantine_path in quarantine_paths:
        alf_path = quarantine_path.joinpath(ssl.session_path.relative_to(one.alyx._par.CACHE_DIR), 'alf', ssl.pname, 'pykilosort')
        if alf_path.exists():
            break
    if not alf_path.exists():
        print(i, rec.pid, 'NO SORTING')
        continue
    print(i, rec.pid, f'making pictures {ssl.session_path}')
    path_figures = path_reveal_figures.joinpath(rec.pid)
    sr = ssl.raw_electrophysiology(band="ap", stream=False)
    spikes_a, clusters_a, channels_a = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    try:
        clusters_a = ssl.merge_clusters(spikes_a, clusters_a, channels_a)
        spikes_b = alfio.load_object(alf_path, 'spikes', attribute=SPIKES_ATTRIBUTES + ['samples'])
        clusters_b = alfio.load_object(alf_path, 'clusters', attribute=CLUSTERS_ATTRIBUTES)
        drift_b = alfio.load_object(alf_path, 'drift')
        # channels_raw = alfio.load_object(alf_path, 'channels')
        clusters_b = ssl.merge_clusters(spikes_b, clusters_b, channels_a)
    except Exception:  # 8b735d77-b77b-4243-8821-37802bf402fe
        print(i, rec.pid, 'REPORT ERROR !! ')
        continue
    # make the reveal website pictures
    path_figures.mkdir(exist_ok=True, parents=True)
    compare.reveal_pid(ssl, spikes_a, spikes_b, clusters_a, clusters_b, channels_a, path_figures, drift_b)

    # save the clusters table for overall statistics
    clusters_b['pid'] = rec.pid
    clusters_b['eid'] = ssl.eid
    pd.DataFrame(clusters_b).to_parquet(path_reveal_figures.joinpath('parquet', f'{rec.pid}_clusters.pqt'))
