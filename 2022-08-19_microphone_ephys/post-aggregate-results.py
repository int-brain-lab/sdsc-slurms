# source ~/Documents/PYTHON/envs/ibllib/bin/activate
from pathlib import Path
import numpy as np
import pandas as pd

APP_PATH = Path.home().joinpath('ceph', 'microphone')
files_welch = APP_PATH.rglob('welchogram.npz')


## %%
records = []
for file_welch in files_welch:
    fw = np.load(file_welch)
    W, tscale, fscale, detect = [fw[k] for k in ['arr_0', 'arr_1', 'arr_2', 'arr_3']]
    subject, date, number = file_welch.parts[-2].split('__')
    rec = {'subject': subject, 'date': date, 'number': number, 'events': detect.size, 'psd': np.mean(np.sum(W, axis=-1))}
    records.append(rec)


df = pd.DataFrame(records)
df.to_parquet(APP_PATH.joinpath('microphone.pqt'))