from pathlib import Path
import pandas as pd
import numpy as np

from deploy.iblsdsc import OneSdsc as ONE

from brainbox.io.one import SpikeSortingLoader

pid = '0654f7ee-21ba-49e2-af8c-50c1e23e85ed'

one = ONE()
ssl = SpikeSortingLoader(one=one, pid=pid)
spikes, clusters, channels = ssl.load_spike_sorting()


# %%
from pathlib import Path
import shutil
out_path = Path('/mnt/home/owinter/ceph/ea/cells')

for error_file in out_path.glob('*.error'):
    print(error_file)
    shutil.rmtree(error_file.joinpath(error_file.stem))
    error_file.unlink()
