"""
One-off bad-channel detection for the LFP compression QC.

The compression pipeline only runs ``detect_bad_channels_cbin`` when a PID has no
Cadzow checkpoint yet.  Because the SDSC job seeds nearly every PID from an archived
``lf_resampled_car_cadzow.npy``, detection is skipped on those reruns and no bad-channel
``labels`` reach the archive.  This script recomputes the labels once from the raw ``.cbin``
(the only stage that can see per-channel quality) and writes them per-PID as
``<pid>/channel_labels.npy``.  ``compress.py`` then loads that array and feeds it into
``compress_bin_to_h5`` on the next (re)compression, so the ``labels`` attr is written even
when resuming from a checkpoint.

Run this BEFORE recompressing.  Output per PID:
    <OUTPUT_ROOT>/<pid>/channel_labels.npy   int8 (nc,)  0=good 1=dead 2=noisy 3=outside

Labels convention (ibldsp.voltage.detect_bad_channels): 0 good, 1 dead, 2 noisy,
3 outside brain.  The presence of channel_labels.npy is the completion sentinel; existing
files are skipped unless --overwrite.
"""
import argparse
import os
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import detect_bad_channels_cbin

os.environ['TQDM_DISABLE'] = '1'

N_OUTER = 8  # detection is lighter than compression; run more PIDs at once

OUTPUT_ROOT = Path(os.environ.get('OUTPUT_ROOT', '/mnt/home/owinter/ceph/ea/denoised_lfp'))
TABLES_DIR  = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
FILE_INSERTIONS = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')

LABELS_FILE = 'channel_labels.npy'


def detect_pid(pid, overwrite=False):
    """Detect bad channels for one PID and write ``<pid>/channel_labels.npy``.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    overwrite : bool
        Recompute even if channel_labels.npy already exists.

    Returns
    -------
    dict or None
        Per-PID label counts, or None on failure.
    """
    out_dir = OUTPUT_ROOT.joinpath(pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_file = out_dir.joinpath(LABELS_FILE)
    if labels_file.exists() and not overwrite:
        labels = np.load(labels_file)
        return _counts(pid, labels)

    err_file = out_dir.joinpath(f'{pid}_labels.error')
    err_file.unlink(missing_ok=True)
    try:
        one = ONE()
        ssl = SpikeSortingLoader(one=one, pid=pid)
        sr = ssl.raw_electrophysiology(band='lf', stream=False)
        labels, _ = detect_bad_channels_cbin(sr, return_features=True)
        labels = labels.astype(np.int8)
        # atomic write so a hard kill never leaves a truncated sentinel
        tmp = labels_file.with_suffix('.npytmp')
        np.save(tmp, labels)
        tmp.rename(labels_file)
        c = _counts(pid, labels)
        print(f"{pid[:8]} labels: {c['n_dead']}d {c['n_noisy']}n {c['n_outside']}o "
              f"/ {c['nc']} channels", flush=True)
        return c
    except Exception:
        tb = traceback.format_exc()
        err_file.write_text(tb)
        print(f'{pid[:8]} ERROR:\n{tb}', flush=True)
        return None


def _counts(pid, labels):
    """Summarise a label array into a one-row-per-PID count dict."""
    labels = np.asarray(labels)
    return dict(
        pid=pid,
        nc=int(labels.size),
        n_good=int(np.sum(labels == 0)),
        n_dead=int(np.sum(labels == 1)),
        n_noisy=int(np.sum(labels == 2)),
        n_outside=int(np.sum(labels == 3)),
        n_bad=int(np.sum(labels != 0)),
    )


def worker_init():
    time.sleep(os.getpid() % 60)  # stagger ONE authentication


parser = argparse.ArgumentParser(description='Precompute bad-channel labels per PID.')
parser.add_argument('--overwrite', action='store_true', help='recompute even if channel_labels.npy exists')
parser.add_argument('--pids', nargs='*', default=None, help='explicit PID list (overrides parquet selection)')
parser.add_argument('--limit', type=int, default=None, help='process at most N PIDs after array slicing')
args = parser.parse_args()

if args.pids:
    pids = args.pids
else:
    df_insertions = pd.read_parquet(FILE_INSERTIONS)
    pids = list(df_insertions.loc[df_insertions['histology'] != '', 'pid'])
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
pids = pids[task_id::n_tasks]
if args.limit is not None:
    pids = pids[:args.limit]
print(f'Task {task_id}/{n_tasks}: detecting bad channels for {len(pids)} PIDs ({N_OUTER} workers)', flush=True)

jobs = [joblib.delayed(detect_pid)(pid=pid, overwrite=args.overwrite) for pid in pids]
results = joblib.Parallel(n_jobs=N_OUTER, backend='loky', initializer=worker_init)(jobs)

rows = [r for r in results if r is not None]
if rows:
    df = pd.DataFrame(rows)
    out = OUTPUT_ROOT.joinpath(f'bad_channel_counts_{task_id:02d}_{n_tasks:02d}.pqt')
    df.to_parquet(out)
    print(f'Wrote {len(df)} rows → {out}', flush=True)
    print(f'  failures: {len(results) - len(rows)}', flush=True)