"""
Reduce step: concatenate per-insertion HDF5 files produced by cells.py into
single arrays and dataframes saved under AGG_PATH.

Expected outputs
----------------
avg_waveforms.npy            (total_nb_traces, ns=82)  float32  – all neighbourhood traces
avg_waveform_peak_channel.npy (n_clusters_total, ns=82) float32  – peak-channel trace per cluster
acgs_log_bins.npy            (n_clusters_total, 128)    float32  – log-binned ACGs
acgs_log_times.npy           (128,)                     float64  – bin centres (shared across insertions)
df_clusters.pqt              (n_clusters_total, ~50 cols)        – cluster table after ssl merge
avg_waveforms_index.pqt      (total_nb_traces, 3 cols)           – pid / cluster_id / abs_channel
avg_waveform_features.pqt    (n_clusters_total, ~15 cols)        – waveform shape features
df_clusters_extended.pqt     (n_clusters_total, 2 cols)          – burstiness, memory

Approximate sizes for ~700 insertions, ~800 clusters each, nc≈40–54 channels, ns=82 samples:
  avg_waveforms              ~(25 M, 82)   ≈ 8 GB
  avg_waveform_peak_channel  ~(560 K, 82)  ≈ 180 MB
  acgs_log_bins              ~(560 K, 128) ≈ 290 MB
  df_clusters                ~560 K rows   ≈ 500 MB parquet
  avg_waveforms_index        ~25 M rows    ≈ 1.5 GB parquet
  avg_waveform_features      ~560 K rows   ≈ 50 MB parquet
  df_clusters_extended       ~560 K rows   ≈ 10 MB parquet
"""

from pathlib import Path
import tqdm
import h5py
import numpy as np
import pandas as pd

CELLS_PATH = Path('/mnt/home/owinter/ceph/ea/cells')
AGG_PATH = Path('/mnt/home/owinter/ceph/ea/cells_aggregates')

h5_files = sorted(CELLS_PATH.rglob('*.h5'))

acc_avg_waveforms = []
acc_avg_waveform_peak_channel = []
acc_acgs_log_bins = []
acc_df_clusters = []
acc_df_avg_waveforms_index = []
acc_df_wf_features = []
acc_df_clusters_extended = []

for fil in tqdm.tqdm(h5_files, desc='reduce'):
    with h5py.File(fil, 'r') as h5:
        acc_avg_waveforms.append(h5['avg_waveforms'][:])
        acc_avg_waveform_peak_channel.append(h5['avg_waveform_peak_channel'][:])
        acc_acgs_log_bins.append(h5['acgs_log_bins'][:])
        acgs_log_times = h5['acgs_log_times'][:]  # identical across insertions; keep last
    acc_df_clusters.append(pd.read_hdf(fil, key='df_clusters'))
    acc_df_avg_waveforms_index.append(pd.read_hdf(fil, key='avg_waveforms_index'))
    acc_df_wf_features.append(pd.read_hdf(fil, key='avg_waveform_features'))
    acc_df_clusters_extended.append(pd.read_hdf(fil, key='df_clusters_extended'))

avg_waveforms = np.concatenate(acc_avg_waveforms, axis=0)
avg_waveform_peak_channel = np.concatenate(acc_avg_waveform_peak_channel, axis=0)
acgs_log_bins = np.concatenate(acc_acgs_log_bins, axis=0)
df_clusters = pd.concat(acc_df_clusters)
df_avg_waveforms_index = pd.concat(acc_df_avg_waveforms_index, ignore_index=True)
df_wf_features = pd.concat(acc_df_wf_features)
df_clusters_extended = pd.concat(acc_df_clusters_extended)

print(f'n_insertions           {len(h5_files)}')
print(f'avg_waveforms          {avg_waveforms.shape}')
print(f'avg_wf_peak_ch         {avg_waveform_peak_channel.shape}')
print(f'acgs_log_bins          {acgs_log_bins.shape}')
print(f'df_clusters            {df_clusters.shape}')
print(f'avg_waveforms_index    {df_avg_waveforms_index.shape}')
print(f'avg_waveform_features  {df_wf_features.shape}')
print(f'df_clusters_extended   {df_clusters_extended.shape}')

AGG_PATH.mkdir(parents=True, exist_ok=True)
np.save(AGG_PATH.joinpath('avg_waveforms.npy'), avg_waveforms)
np.save(AGG_PATH.joinpath('avg_waveform_peak_channel.npy'), avg_waveform_peak_channel)
np.save(AGG_PATH.joinpath('acgs_log_bins.npy'), acgs_log_bins)
np.save(AGG_PATH.joinpath('acgs_log_times.npy'), acgs_log_times)
df_clusters.to_parquet(AGG_PATH.joinpath('df_clusters.pqt'))
df_avg_waveforms_index.to_parquet(AGG_PATH.joinpath('avg_waveforms_index.pqt'))
df_wf_features.to_parquet(AGG_PATH.joinpath('avg_waveform_features.pqt'))
df_clusters_extended.to_parquet(AGG_PATH.joinpath('df_clusters_extended.pqt'))


# n_insertions           1091
# avg_waveforms          (36238663, 128)
# avg_wf_peak_ch         (925251, 128)
# acgs_log_bins          (925251, 128)
# df_clusters            (925251, 35)
# avg_waveforms_index    (36238663, 3)
# avg_waveform_features  (925251, 23)
# df_clusters_extended   (925251, 2)
