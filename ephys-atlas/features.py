import functools
import joblib
from pathlib import Path

import pandas as pd
import numpy as np

import ephys_atlas.features

from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage

root_path = Path('/mnt/home/owinter/ceph/EA')
n_jobs = 48

df_snippets = pd.read_csv(root_path.joinpath('/mnt/home/owinter/Documents/sdsc-slurms/ephys-atlas/snippets_others.csv'))
df_snippets = df_snippets[df_snippets['neuropixel'] != 'NP2.4'].copy()
#df_snippets = df_snippets[np.mod(df_snippets['i_snippet'], 2) == 0].copy()

ns_ap = 2 ** 18  # around 8 seconds of data
lf_offset = 3  # in samples, the shift to apply to LF to align with AP


def compute_snippet_features(subject=None, pid=None, eid=None, pname=None, t0=None, **kwargs):
    one = ONE(mode='local')  # nb: this can't be instantiated in any other mode
    path_snippet = root_path.joinpath(subject, pid, f'T{t0 * 1e3 :010.0f}')
    path_waveforms = path_snippet.joinpath('waveforms')
    path_waveforms.mkdir(parents=True, exist_ok=True)
    file_csd = path_snippet.joinpath('csd.pqt')
    file_lf = path_snippet.joinpath('lf.pqt')
    file_ap = path_snippet.joinpath('ap.pqt')
    file_wav = path_snippet.joinpath('wav.pqt')
    file_channels = path_snippet.joinpath('channels.pqt')

    ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
    channels = ssl.load_channels()
    sr_ap = ssl.raw_electrophysiology(band='ap', stream=False)  # TODO stream false
    sr_lf = ssl.raw_electrophysiology(band='lf', stream=False)
    ns_lf = int(ns_ap * sr_lf.fs / sr_ap.fs)

    @functools.lru_cache(maxsize=1)
    def destripe_ap(t0):
        raw_ap = sr_ap[slice(n0 := int(sr_ap.fs * t0), n0 + ns_ap), :-sr_ap.nsync].T
        return ibldsp.voltage.destripe(
            raw_ap, fs=sr_ap.fs, neuropixel_version=sr_ap.major_version, channel_labels=channels['labels'],
            k_filter=False,
        )

    @functools.lru_cache(maxsize=1)
    def destripe_lf(t0):
        raw_lf = sr_lf[slice(n0 := int(sr_lf.fs * t0 + lf_offset), n0 + ns_lf), :-sr_lf.nsync].T
        return ibldsp.voltage.destripe_lfp(
            raw_lf, fs=sr_lf.fs, channel_labels=channels['labels'],
        )

    if not file_channels.exists():
        df_channels = pd.DataFrame(channels).rename(columns={'rawInd': 'channel'})
        df_channels.to_parquet(file_channels)

    if not file_lf.exists():
        des_lf = destripe_lf(t0)
        df_lf = ephys_atlas.features.lf(des_lf, fs=sr_lf.fs)
        df_lf.to_parquet(file_lf)

    if not file_csd.exists():
        des_lf = destripe_lf(t0)
        df_csd = ephys_atlas.features.csd(des_lf, fs=sr_lf.fs, geometry=sr_ap.geometry, decimate=10)
        df_csd.to_parquet(file_csd)

    if not file_ap.exists():
        des_ap = destripe_ap(t0)
        df_ap = ephys_atlas.features.ap(des_ap, geometry=sr_ap.geometry)
        df_ap.to_parquet(file_ap)

    if not file_wav.exists():
        des_ap = destripe_ap(t0)
        df_waveforms, waveforms = ephys_atlas.features.spikes(des_ap, fs=sr_ap.fs, geometry=sr_ap.geometry)
        df_waveforms.to_parquet(file_wav)
        np.save(path_waveforms.joinpath('raw.npy'), waveforms['raw'].astype(np.float16))
        np.save(path_waveforms.joinpath('denoised.npy'), waveforms['denoised'].astype(np.float16))
        np.save(path_waveforms.joinpath('waveform_channels.npy'), waveforms['channel_index'])
        waveforms['df_spikes'].to_parquet(path_waveforms.joinpath('spikes.pqt'))
        # waveforms = {
        #     'raw': np.load(path_waveforms.joinpath('raw.npy')).astype(np.float32),
        #     'denoised': np.load(path_waveforms.joinpath('denoised.npy')).astype(np.float32),
        #     'channel_index': np.load(path_waveforms.joinpath('waveform_channels.npy')),
        #     'df_spikes': pd.read_parquet(path_waveforms.joinpath('spikes.pqt')),
        # }

jobs = [joblib.delayed(compute_snippet_features)(**dict(args)) for _, args in df_snippets.iterrows()]
joblib.Parallel(n_jobs=n_jobs)(jobs)

