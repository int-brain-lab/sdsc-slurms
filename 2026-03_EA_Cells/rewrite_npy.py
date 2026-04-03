from pathlib import Path
import numpy as np
from one.api import ONE
from deploy.iblsdsc import OneSdsc
from brainbox.io.one import SpikeSortingLoader

WORKDIR = Path('/mnt/home/owinter/ceph/ea/cells')
# WORKDIR = Path.home().joinpath('scratch/lfp')
TABLES_DIR = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
# oneloc = OneSdsc(mode='local', tables_dir=TABLES_DIR)
one = OneSdsc()

def rewrite_lfp(file_lfp, nc):
    ns = file_lfp.stat().st_size / 2 / nc
    assert ns % 1 == 0
    a = np.memmap(file_lfp, dtype='float16', mode='r', shape=(int(ns), nc))
    np.save(file_lfp.with_suffix('.npy'), a)
    file_lfp.unlink()

# %%
import joblib
import tqdm
jobs = []

def job_generator():
    for file_lfp in WORKDIR.rglob("lf_resampled.bin"):
        pid = file_lfp.parts[-2]
        ssl = SpikeSortingLoader(one=one, pid=pid)
        sr = ssl.raw_electrophysiology(band="lf", stream=False)
        yield joblib.delayed(rewrite_lfp)(file_lfp, nc=sr.nc - sr.nsync)

joblib.Parallel(n_jobs=20)(job_generator())


