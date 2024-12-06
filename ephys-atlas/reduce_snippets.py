"""
Reduce the features to a single parquet file per subject, for downloading
"""
from functools import reduce
from pathlib import Path

import scipy.signal
import pandas as pd

from one.api import ONE
from iblutil.numerical import hash_uuids
from iblutil.io import hashfile

# rsync -av popeye:/mnt/home/owinter/ceph/EA/ /home/olivier/Documents/2024/EA_recuced --exclude '*.npy*'
feature_names = ['ap', 'lf', 'csd', 'wav']
root_path = Path('/mnt/home/owinter/ceph/EA')
out_path = Path('/mnt/home/owinter/ceph/EA_reduced')
subjects = sorted([p.name for p in root_path.glob('*') if p.is_dir()])
OVERWRITE = True

def _compute_hash(snippet_path):
    return hash_uuids([hashfile.md5(f) for f in snippet_path.glob('*.pqt')], 'md5')

def has_snippet_changed(snippet_path):
    file_hash = next(snippet_path.glob("*.md5"), None)
    if file_hash is None:
        return True
    if file_hash.name == _compute_hash(snippet_path):
        return False
    return True
    
def save_hash(snippet_path):
    hash = _compute_hash(snippet_path)
    for hash_file in snippet_path.glob('*.md5'):
        hash_file.unlink()
    snippet_path.joinpath(f"{hash}.md5").touch()


for i, subject in enumerate(subjects):
    output_file = out_path.joinpath(f'{subject}.pqt')
    if output_file.exists() and not OVERWRITE:
        continue
    print(i, len(subjects), subject)
    snippet_paths = [sn.parent for sn in root_path.joinpath(subject).rglob('wav.pqt')]
    if len(snippet_paths) == 0:
        print(subject, 'no waveform file found')
        continue
    df_voltage, df_channels = ([], [])
    for snippet_path in snippet_paths:
        if not has_snippet_changed(snippet_path):
            continue
        t0, pid = (float(snippet_path.parts[-1][1:]) / 1e3 , snippet_path.parts[-2])
        df_snippet = {fn: None for fn in feature_names}
        for fn in feature_names:
            df_snippet[fn] = pd.read_parquet(snippet_path.joinpath(f'{fn}.pqt'))            
        dfv = reduce(lambda left, right: pd.merge(left, right, on='channel', how='outer'),
                     [df_snippet[fn] for fn in feature_names])
        dfc = pd.read_parquet(snippet_path.joinpath('channels.pqt'))
        dfv['pid'] = pid
        dfv['t0'] = t0
        dfc['pid'] = pid
        dfc['t0'] = t0
        df_channels.append(dfc)
        df_voltage.append(dfv)
        save_hash(snippet_path)
    pd.concat(df_voltage).to_parquet(output_file)
    pd.concat(df_channels).to_parquet(out_path.joinpath(f'{subject}_channels.pqt'))
