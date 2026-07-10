from pathlib import Path
import pandas as pd
import numpy as np

from one.api import ONE

from brainbox.io.one import SpikeSortingLoader
import ephysatlas.cells

pid = '0fed7207-f747-428b-b4c0-854cabb50d9e'

one = ONE()
ssl = SpikeSortingLoader(one=one, pid=pid)
spikes, clusters, channels = ssl.load_spike_sorting()

df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

df_clusters, stpc, tscale, coupling_strength, taper, coupling_delay, firing_rates = (
    ephysatlas.cells.spike_triggered_population_coupling(
        spikes,
        df_clusters,
    )
)


raise NotImplementedError
# %% First pass of error fixing

ERRORS = [
    '0654f7ee-21ba-49e2-af8c-50c1e23e85ed',  # Fixme md5 mismatch
    '0fed7207-f747-428b-b4c0-854cabb50d9e',  # Invalid entry in coordinate array
    '11a5a93e-58a9-4ed0-995e-52279ec16b98',  # Invalid entry in coordinate array
    '1496383d-424d-4999-8e28-fec1eea7e2d6',  # alf object not found
    '3d496c9d-0cb1-40cd-be08-2de0f6bf108f',  # alf object not found
    '4138aef6-930d-4cdb-9ec1-7b032e59766b',  # alf object not found
    '50f1512d-dd41-4a0c-b3ab-b0564f0424d7',  # Invalid entry in coordinate array
    '5a34d971-1cb3-4f0e-8dfe-e51e2313a668',  # Invalid entry in coordinate array
    '6e61f777-90b9-4656-af30-9ed54d426c5a',  # index 0 is out of bounds channel regions
    '78419cd7-0d9e-4b91-86ca-0572c32c2332',  # alf object not found
    '7fe94f3c-69f9-45fe-9de0-66cf306d4d48',
    '8e2603d7-4631-499c-930e-b8fe4aa1f8ed',
    '9328cf49-2093-428e-9c99-d4d9c189ae63',
    '9ce950c3-e9cc-4669-a659-4b1b77db5f37',
    '9edc5d65-cb26-475c-8544-e6d1dacae41d',
    'a3f6b2cf-2781-4ef2-9894-4b7ac6af9b23',
    'b89b9d2e-aa6c-42de-ad49-d3ec0f6c689f',
    'ce16c71a-f0a6-48b7-bc2f-430ff94df5de',
    'd4507b91-a2b1-4a8b-b723-1b245054aff2',
    'df3adc27-699f-40a1-a70b-c6e9d498dd9e',
    'e7450825-6838-4336-9b26-1f2f6cf81ec1',
    'eedba3b6-b70a-4ea7-a87a-79f6ce7403e9',
    'f6a89dc9-acfd-414a-a78b-3a21300bca03',
    ]

# %%
import boto3
import subprocess

session = boto3.Session(profile_name="ibl")
s3 = session.client("s3")

def get_size_s3_file(s3_uri: str) -> None:
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")
    # Strip s3:// and split into bucket + key
    without_scheme = s3_uri[5:]
    bucket, key = without_scheme.split("/", 1)
    resp = s3.head_object(Bucket=bucket, Key=key)
    return resp



one = ONE()
ERRORS = [    'eedba3b6-b70a-4ea7-a87a-79f6ce7403e9',
    'f6a89dc9-acfd-414a-a78b-3a21300bca03',]
for pid in ERRORS:
    ssl = SpikeSortingLoader(one=one, pid=pid)
    collection = ssl._get_spike_sorting_collection()
    df_dsets = one.list_datasets(eid=ssl.eid, collection=collection, details=True)

    for did, rec in df_dsets.iterrows():
        local_file = ssl.session_path.joinpath(rec.rel_path)
        relative_path_uuid = local_file.relative_to('/Users/olivier/Downloads/ONE/alyx.internationalbrainlab.org')
        relative_path_uuid = relative_path_uuid.with_suffix(f'.{did}{relative_path_uuid.suffix}')
        remote_file_sdsc = Path('/mnt/ibl').joinpath(relative_path_uuid)
        s3_uri = "s3://ibl-brain-wide-map-private/data/" + str(relative_path_uuid)

        try:
            result = subprocess.run(
                [f"ssh sdsc md5sum {remote_file_sdsc}"],
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            md5sdsc = result.stdout.split(' ')[0]
            response = get_size_s3_file(s3_uri)
            print(local_file)
            print(f'{rec['hash']} md5, size: {rec.file_size}  alyx')
            print(f"{response['ETag'].replace('"', '')} md5, size: {response["ContentLength"]}  s3")
            print(md5sdsc, 'md5 sdsc')

            if rec['hash'] != md5sdsc:
                print('patching dataset')
                one.alyx.rest('datasets', 'partial_update', id=did, data={'hash': md5sdsc})


        except subprocess.CalledProcessError as e:
            print('ERROR', local_file)
            print(f"Error: {e.returncode} {e.output}")



# %%
file_insertions = Path('/Users/olivier/Documents/datadisk/paper-ephys-atlas/s3/project-metadata/df_probe_details_ibl_neuropixel_brainwide_01.pqt')
TABLES_DIR = Path('/Users/olivier/Documents/datadisk/paper-ephys-atlas/s3/project-metadata/one_cache-ibl_neuropixel_brainwide_01')
df_insertions = pd.read_parquet(file_insertions)
np.sum(df_insertions['histology'] != '')

# rsync -av --progress -e ssh /Users/olivier/Documents/datadisk/paper-ephys-atlas/s3/project-metadata/df_probe_details_ibl_neuropixel_brainwide_01.pqt popeye:/mnt/home/owinter/Documents/cache_tables



# %%
from pathlib import Path
import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

one = ONE()
WORKDIR = Path('/mnt/home/owinter/ceph/ea/denoised_lfp')
WORKDIR = Path.home().joinpath('scratch/lfp')


for file_lfp in WORKDIR.rglob('lf_resampled.bin'):
    print(file_lfp)
    pid = file_lfp.parts[-2]
    ssl = SpikeSortingLoader(one=one, pid=pid)
    sr = ssl.raw_electrophysiology(band='lf', stream=False)
    ns = file_lfp.stat().st_size / 2 / (sr.nc - sr.nsync)
    assert ns % 1 == 0
    a = np.memmap(file_lfp, dtype='float16', mode='r', shape=(int(ns), sr.nc - sr.nsync))
    np.save(file_lfp.with_suffix('.npy'), a)
    file_lfp.unlink()
    break

# %%
import spikeglx
from viewephys.gui import viewephys
sr = spikeglx.Reader('/Users/olivier/scratch/lfp/1e104bf4-7a24-4624-a5b2-c2c8289c0de7/lf_resampled.bin', nc=384, ns=ns, fs=250, dtype=np.float16)
srr = spikeglx.Reader('/Users/olivier/scratch/lfp/1e104bf4-7a24-4624-a5b2-c2c8289c0de7/lf_resampled.npy', fs=250)

np.testing.assert_array_equal(sr[:100, :], srr[:100, :])

sr
