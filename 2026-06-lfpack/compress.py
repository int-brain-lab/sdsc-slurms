"""
LFP compression cluster job — runs on a single 48-core node.

Parallelism strategy: 4 outer workers × 12 inner cores = 48 cores total.
- Outer (joblib loky): 4 PIDs processed concurrently.
- Inner (ProcessPoolExecutor / joblib): each compress_bin_to_h5 call uses 12 cores
  for both the Cadzow decimation stage and the SVD+WP compression stage.

The Cadzow checkpoint (~1.4 GB/PID) lives on /scratch (fast local NVMe) and is
shared between the default and aggressive compression passes, so the expensive
Cadzow step runs only once per PID.  The tiny output H5s (~2 MB each) are written
directly to ceph.  Scratch is cleaned up per-PID regardless of success or failure.
"""
import argparse
import os
import shutil
import time
import traceback
from pathlib import Path

import joblib
import pandas as pd

from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader
from lfpack import compress_bin_to_h5

os.environ['TQDM_DISABLE'] = '1'

# ── Parallelism ────────────────────────────────────────────────────────────────
N_OUTER = 4   # PIDs processed simultaneously
N_INNER = 12  # cores per PID  (N_OUTER × N_INNER == 48)

# ── Compression parameters (mirrors 2026-06-02_LFP_compression.py) ────────────
Q = 10
PARAMS = {
    'default':    dict(epsilon=150.0, alpha=28.0),
    'aggressive': dict(epsilon=450.0, alpha=96.0),
}
CADZOW_KWARGS = dict(rank=5, niter=1, fmax=None, nswx=64, ovx=32, gap_threshold=2.0, ppca_k=2.0)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRATCH_ROOT = Path(os.environ.get('SCRATCH_ROOT', '/tmp/lfpack_local'))
OUTPUT_ROOT  = Path('/mnt/home/owinter/ceph/ea/cells')
TABLES_DIR   = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
FILE_INSERTIONS = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')


def compress_pid(pid, overwrite=False):
    """Compress one PID: Cadzow checkpoint on scratch → default H5 → aggressive H5 on ceph.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    overwrite : bool
        Delete existing H5 outputs and recompute from scratch.
    """
    out_dir   = OUTPUT_ROOT.joinpath(pid)

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = SCRATCH_ROOT.joinpath(pid)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    files = {
        'default':    out_dir.joinpath('lf_compressed.h5'),
        'aggressive': out_dir.joinpath('lf_compressed_aggressive.h5'),
    }
    if files['default'].exists() and files['aggressive'].exists() and not overwrite:
        return

    if overwrite:
        for f in files.values():
            f.unlink(missing_ok=True)

    # Cadzow checkpoint: fast local NVMe during computation, archived to ceph afterwards.
    # If the ceph archive exists from a prior run, seed scratch from it to skip recomputation.
    cadzow_scratch  = scratch_dir.joinpath('lf_resampled_car_cadzow.npy')
    cadzow_archive  = out_dir.joinpath('lf_resampled_car_cadzow.npy')
    if cadzow_archive.exists() and not cadzow_scratch.exists():
        shutil.copy2(cadzow_archive, cadzow_scratch)
        print(f'{pid[:8]} Cadzow: seeded from ceph archive', flush=True)

    try:
        one = ONE()
        ssl = SpikeSortingLoader(one=one, pid=pid)
        sr  = ssl.raw_electrophysiology(band='lf', stream=False)

        for lbl, params in PARAMS.items():
            if files[lbl].exists():
                print(f'{pid[:8]} {lbl}: exists, skipping', flush=True)
                continue
            print(f'{pid[:8]} {lbl}: compressing …', flush=True)
            t0 = time.perf_counter()
            compress_bin_to_h5(
                sr.file_bin, files[lbl],
                q=Q,
                cadzow_checkpoint_file=cadzow_scratch,
                cadzow_kwargs=CADZOW_KWARGS,
                n_jobs=N_INNER,
                **params,
            )
            print(f'{pid[:8]} {lbl}: done in {time.perf_counter() - t0:.1f} s', flush=True)

        if cadzow_scratch.exists() and not cadzow_archive.exists():
            shutil.copy2(cadzow_scratch, cadzow_archive)
            print(f'{pid[:8]} Cadzow: archived to ceph', flush=True)
        out_dir.joinpath(f'{pid}_compress.error').unlink(missing_ok=True)

    except Exception:
        tb = traceback.format_exc()
        out_dir.joinpath(f'{pid}_compress.error').write_text(tb)
        print(f'{pid[:8]} ERROR:\n{tb}', flush=True)

    finally:
        # Always clean up scratch for this PID to free space for the next job.
        shutil.rmtree(scratch_dir, ignore_errors=True)


def worker_init():
    # Stagger worker startup to avoid thundering-herd on ONE authentication.
    time.sleep(os.getpid() % 60)


parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', help='recompute even if .done already exists')
args = parser.parse_args()

df_insertions = pd.read_parquet(FILE_INSERTIONS)
pids = list(df_insertions.loc[df_insertions['histology'] != '', 'pid'])
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
pids = pids[task_id::n_tasks]
print(f'Task {task_id}/{n_tasks}: queuing {len(pids)} PIDs  ({N_OUTER} outer × {N_INNER} inner cores)', flush=True)

jobs = [joblib.delayed(compress_pid)(pid=pid, overwrite=args.overwrite) for pid in pids]
joblib.Parallel(n_jobs=N_OUTER, backend='loky', initializer=worker_init)(jobs)
