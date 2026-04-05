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


# Each folder is named by the pid and contains:
#     clusters.pqt
#     lf_resampled.npy
#     stlfp.npy
#     stpc.npy
#     stpc.png

EXCLUDES = [
    '1122230f-42b8-45f9-ad7e-00c27ae087c8',  # dartsort error
    '1dd218c9-ac97-4d91-80d0-a8a660bf7395',  # dartsort broadcast
    '316a733a-5358-4d1d-9f7f-179ba3c90adf',  # dartsort error
    '71a92c54-69f0-488b-ae2a-cb6c1524233c',  # dartsort error
    '80494687-eb74-43c6-801c-e99fd6621d51',  # dartsort broadcast
    'fb76fd5c-0b91-41f2-9b94-0f64b62396cb',  # dartsort broadcast
    'ce16c71a-f0a6-48b7-bc2f-430ff94df5de',  # spike sorting stuck Elbocal
]

for folder_pid in out_path.glob('*'):
    if not folder_pid.is_dir():
        continue
    pid = folder_pid.parts[-1]
    if pid in EXCLUDES:
        continue
    if next(folder_pid.glob('stlfp.npy'), None) is None:
        print(pid, 'stlfp.npy missing')
    if next(folder_pid.glob('stpc.npy'), None) is None:
        print(pid, 'stpc.npy missing')


# 24cb326a-e04d-4a18-a96d-1edef60cc40b stlfp.npy missing
# ce16c71a-f0a6-48b7-bc2f-430ff94df5de stlfp.npy missing
# ce16c71a-f0a6-48b7-bc2f-430ff94df5de stpc.npy missing