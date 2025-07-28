import os
os.environ['TQDM_DISABLE'] = '1'
import argparse
import joblib
from pathlib import Path
import shutil

import numpy as np
from deploy.iblsdsc import OneSdsc as ONE
import ibldsp.voltage
from brainbox.io.one import SpikeSortingLoader
from brainwidemap.bwm_loading import bwm_query

n_jobs = 48 // 4
output_path = Path('/mnt/home/owinter/ceph/bwm/saturation')
one = ONE(mode='local')
df_pids = bwm_query(one=one)

def compute_saturation_pid(rec):
    output_file = output_path.joinpath(rec.pid, '_iblqc_ephysSaturation.samples.npy')
    if output_file.exists():
        print(rec.pid, 'already done')
        return
    ssl = SpikeSortingLoader(eid=rec.eid, one=one, pname=rec.probe_name)
    sr = ssl.raw_electrophysiology(band='ap', stream=False)
    output_file_tmp = Path(f'/scratch/{rec.pid}/_iblqc_ephysSaturation.samples.npy')
    output_file_tmp.parent.mkdir(parents=True, exist_ok=True)
    ibldsp.voltage.saturation_cbin(sr, file_saturation=output_file_tmp, n_jobs=4)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(output_file_tmp, output_file)
    print(rec.pid, 'done')

jobs = [joblib.delayed(compute_saturation_pid)(rec=rec) for _, rec in df_pids.iterrows()]
joblib.Parallel(n_jobs=n_jobs)(jobs)
