import os
os.environ['TQDM_DISABLE'] = '1'

import joblib
from pathlib import Path
import shutil

from deploy.iblsdsc import OneSdsc as ONE

import ibldsp.waveform_extraction
from brainbox.io.one import SpikeSortingLoader
from brainwidemap.bwm_loading import bwm_query

REVISION = '2024-01-01'
n_jobs = 13
output_path = Path(f'/mnt/home/owinter/ceph/bwm/waveforms_benchmark/{REVISION}')
output_path.mkdir(parents=True, exist_ok=True)
one = ONE(mode='local')
df_pids = bwm_query(one=one).set_index('pid')
benchmark_pids = [
    "1a276285-8b0e-4cc9-9f0a-a3a002978724",
    "1e104bf4-7a24-4624-a5b2-c2c8289c0de7",
    "5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e",
    "5f7766ce-8e2e-410c-9195-6bf089fea4fd",
    "6638cfb3-3831-4fc2-9327-194b76cf22e1",  # 4: olfactory bulb
    "749cb2b7-e57e-4453-a794-f6230e4d0226",
    "d7ec0892-0a6c-4f4f-9d8f-72083692af5c",
    "da8dfec1-d265-44e8-84ce-6ae9c109b8bd",  # 8: striatum / CP
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0",
    "dc7e9403-19f7-409f-9240-05ee57cb7aea",
    "e8f9fba4-d151-4b00-bee7-447f0f3e752c",
    "eebcaf65-7fa4-4118-869d-a084e84530e2",
    "fe380793-8035-414e-b000-09bfe5ece92a",
]


def compute_waveforms_pid(pid):
    print(pid)
    rec = df_pids.loc[pid, :]
    ssl = SpikeSortingLoader(eid=rec.eid, one=one, pname=rec.probe_name)
    spikes, clusters, channels = ssl.load_spike_sorting(revision=REVISION, dataset_types=['spikes.samples'])
    sr = ssl.raw_electrophysiology(band='ap', stream=False)
    output_path_tmp = Path(f'/scratch/{pid}')
    output_path_tmp.mkdir(parents=True, exist_ok=True)
    ibldsp.waveform_extraction.extract_wfs_cbin(
        bin_file=sr.file_bin,
        output_dir=output_path_tmp,
        spike_samples=spikes['samples'],
        spike_clusters=spikes['clusters'],
        spike_channels=clusters['channels'][spikes['clusters']],
        h=sr.geometry,
        channel_labels=channels['labels'] if 'labels' in channels else None,
        max_wf=256,
        trough_offset=42,
        spike_length_samples=128,
        chunksize_samples=int(30_000),
        n_jobs=3,
        scratch_dir=output_path_tmp.joinpath('.tmp'),
    )
    shutil.copytree(output_path_tmp, output_path.joinpath(pid))
    print(pid, 'done')


jobs = [joblib.delayed(compute_waveforms_pid)(pid=pid) for pid in benchmark_pids]
joblib.Parallel(n_jobs=n_jobs)(jobs)
