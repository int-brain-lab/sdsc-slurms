from pathlib import Path
import pandas as pd
import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

# %%
one = ONE()

pid = '23c9ddcb-31b7-4517-a8f4-ef40adba2ca8'
ssl = SpikeSortingLoader(one=one, pid=pid)
spikes, clusters, channels = ssl.load_spike_sorting()
df_clusters = pd.DataFrame(ssl.merge_clusters(spikes, clusters, channels))

# %%
file_insertions = Path('/Users/olivier/Documents/datadisk/paper-ephys-atlas/s3/project-metadata/df_probe_details_ibl_neuropixel_brainwide_01.pqt')
TABLES_DIR = Path('/Users/olivier/Documents/datadisk/paper-ephys-atlas/s3/project-metadata/one_cache-ibl_neuropixel_brainwide_01')
df_insertions = pd.read_parquet(file_insertions)
np.sum(df_insertions['histology'] != '')


# rsync -av --progress -e ssh /Users/olivier/Documents/datadisk/paper-ephys-atlas/s3/project-metadata/df_probe_details_ibl_neuropixel_brainwide_01.pqt popeye:/mnt/home/owinter/Documents/cache_tables
