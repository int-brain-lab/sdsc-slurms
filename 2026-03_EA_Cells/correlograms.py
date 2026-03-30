from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import traceback

import phylib.stats
from deploy.iblsdsc import OneSdsc
from brainbox.io.one import SpikeSortingLoader

# we instantiate in remote mode to make sure the tables are empty
cache_path = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
output_path = Path('/mnt/sdceph/users/owinter/nemo')
df_pids = pd.read_parquet('/mnt/home/owinter/Documents/cache_tables/df_probe_details_ibl_neuropixel_brainwide_01.pqt')
df_pids = df_pids.loc[df_pids['histology'] != '', :]

one = OneSdsc(mode='local', tables_dir=cache_path)


# %%
def correlograms(rec, one):
    pid_path = output_path.joinpath(rec.pid)
    pid_path.mkdir(parents=True, exist_ok=True)

    AP_SAMPLING_RATE = 30_000
    ssl = SpikeSortingLoader(one=one, eid=rec.eid, pname=rec.pname, spike_sorter='iblsorter')
    ssl.load_channels()
    info = {
        'collection': ssl._get_spike_sorting_collection(),
        'version': ssl.get_version(),
        'histology': ssl.histology
    }
    with open(pid_path.joinpath( 'info.yaml'), 'w') as f:
        yaml.dump(info, f)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    if spikes == clusters == channels == {}:
        raise ValueError('No spike sorting found for this session')
    # here the channels object comes in two flavours: raw channels ('localCoordinates', 'rawInd')
    # and processed channels ('x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um', 'lateral_um')

    if 'localCoordinates' in channels:
        channels = dict(axial_um=channels['localCoordinates'][:, 1], lateral_um=channels['localCoordinates'][:, 0])
    # compute the correlograms
    corr_bin_ts_secs, corr_win_ts_secs = (0.001, 1)  # time-scale long autocorrelogram
    corr_bin_rf_secs, corr_win_rf_secs = (1 / AP_SAMPLING_RATE, .02)  # refractory period short range
    ns_ts = int(np.ceil((corr_win_ts_secs / corr_bin_ts_secs + 1) / 2))
    correlograms_ts = np.zeros((ns_ts, clusters['uuids'].size), dtype=np.int32)
    ns_rf = int(np.ceil((corr_win_rf_secs / corr_bin_rf_secs + 1) / 2))
    correlograms_rf = np.zeros((ns_rf, clusters['uuids'].size), dtype=np.int32)
    for c, s in pd.DataFrame(spikes).groupby('clusters'):
        correlograms_ts[:, c] = phylib.stats.correlograms(
            s['times'], s['clusters'], c, sample_rate=AP_SAMPLING_RATE,
            bin_size=corr_bin_ts_secs, window_size=corr_win_ts_secs, symmetrize=False
        )
        correlograms_rf[:, c] = phylib.stats.correlograms(
            s['times'], s['clusters'], c, sample_rate=AP_SAMPLING_RATE,
            bin_size=corr_bin_rf_secs, window_size=corr_win_rf_secs, symmetrize=False
        )
    df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))
    df_clusters['pid'] = rec.pid
    # the concat syntax sets a higher level index on the dataframe as pid
    np.save(output_path.joinpath(rec.pid, 'correlograms_time_scale.npy'), correlograms_ts)
    np.save(output_path.joinpath(rec.pid, 'correlograms_refractory_period.npy'), correlograms_rf)
    df_clusters.to_parquet(output_path.joinpath(rec.pid, 'clusters.pqt'))


def map_fcn(rec, one):
    try:
        correlograms(rec, one)
    except Exception as e:
        pid_path = output_path.joinpath(rec.pid)
        with open(pid_path.joinpath('error.log'), 'w') as f:
            f.write(traceback.format_exc())

# %%
import joblib

jobs = []

for _, rec in df_pids.iterrows():
    jobs.append(joblib.delayed(map_fcn)(rec=rec, one=one))
    # map_fcn(eid=rec.eid, pname=rec.pname, pid=rec.pid)

joblib.Parallel(n_jobs=48)(jobs)
