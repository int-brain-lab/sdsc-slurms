"""
LFP compression cluster job — runs on a single 48-core node.

Parallelism strategy: 4 outer workers × 12 inner cores = 48 cores total.
- Outer (joblib loky): 4 PIDs processed concurrently.
- Inner (ProcessPoolExecutor / joblib): each compress_bin_to_h5 call uses 12 cores
  for both the Cadzow decimation stage and the SVD+WP compression stage.

The Cadzow checkpoint (~1.4 GB/PID) lives on node-local scratch (SCRATCH_ROOT,
set by the caller -- e.g. compress.sbatch's SCRATCH_BASE=/tmp) and is shared
across every compression tier in PARAMS, so the expensive Cadzow step runs
only once per PID.  The tiny output H5s (~2 MB each) are written directly to
ceph.  Scratch is cleaned up per-PID regardless of success or failure.
"""
import argparse
import os
import shutil
import time
import traceback
from pathlib import Path

import h5py
import joblib
import numpy as np
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
    'mild':       dict(epsilon=100.0, alpha=14.0),
    'default':    dict(epsilon=150.0, alpha=28.0),
    'aggressive': dict(epsilon=450.0, alpha=96.0),
}
CADZOW_KWARGS = dict(rank=5, niter=1, fmax=None, nswx=64, ovx=32, gap_threshold=2.0, ppca_k=2.0)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRATCH_ROOT = Path(os.environ.get('SCRATCH_ROOT', '/tmp/lfpack_local'))
OUTPUT_ROOT  = Path(os.environ.get('OUTPUT_ROOT', '/mnt/home/owinter/ceph/ea/denoised_lfp'))
TABLES_DIR   = Path('/mnt/home/owinter/Documents/cache_tables/one_cache-ibl_neuropixel_brainwide_01')
FILE_INSERTIONS = TABLES_DIR.parent.joinpath('df_probe_details_ibl_neuropixel_brainwide_01.pqt')


def _fix_muted_attr(h5_file):
    """Force the saturation table's ``muted`` attr to True.

    ``compress_bin_to_h5`` sets ``muted = not checkpoint_existed`` — an honest
    self-report of whether *this call* ran the mute step. That undercounts here:
    ``compress_pid`` always deletes a stale pre-mute-fix Cadzow checkpoint on
    ``--overwrite`` (see the comment above ``cadzow_archive.unlink()``), so any
    checkpoint this driver reuses is guaranteed to already carry the late
    post-Cadzow re-mute. Resumed runs would otherwise report ``muted=False``
    despite the data being correctly muted.

    Parameters
    ----------
    h5_file : path-like
        Single-recording HDF5 file just written by ``compress_bin_to_h5``.
    """
    with h5py.File(h5_file, "r+") as f:
        recording = next(iter(f.keys()))
        key = f"{recording}/saturation"
        if key in f:
            f[key].attrs["muted"] = True


def compress_pid(pid, overwrite=False):
    """Compress one PID: Cadzow checkpoint on scratch → one H5 per ``PARAMS`` tier on ceph.

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
        'mild':       out_dir.joinpath('lf_compressed_mild.h5'),
        'default':    out_dir.joinpath('lf_compressed.h5'),
        'aggressive': out_dir.joinpath('lf_compressed_aggressive.h5'),
    }
    # Completion sentinel: every tier's file must exist. Each is written via
    # atomic rename (*.h5tmp → *.h5, see the loop below) so a hard kill never
    # leaves a half-written file that looks done; checking all of them (rather
    # than trusting one hardcoded "last" tier) keeps this correct regardless of
    # PARAMS insertion order or how many tiers exist -- e.g. a PID whose
    # default/aggressive already completed under an older PARAMS still resumes
    # into computing just the newly-added 'mild' tier below.
    if all(f.exists() for f in files.values()) and not overwrite:
        return

    if overwrite:
        for f in files.values():
            f.unlink(missing_ok=True)

    # Remove any stale tmp files left by a prior interrupted run.
    for f in files.values():
        f.with_suffix('.h5tmp').unlink(missing_ok=True)

    # Cadzow checkpoint: fast local NVMe during computation, archived to ceph afterwards.
    # If the ceph archive exists from a prior run, seed scratch from it to skip recomputation.
    cadzow_scratch  = scratch_dir.joinpath('lf_resampled_car_cadzow.npy')
    cadzow_archive  = out_dir.joinpath('lf_resampled_car_cadzow.npy')
    if overwrite:
        # Stage 1 (decimation + saturation muting) must recompute from the raw .cbin,
        # so drop the stale pre-muting checkpoint rather than seeding from it.
        cadzow_archive.unlink(missing_ok=True)
    if cadzow_archive.exists() and not cadzow_scratch.exists():
        shutil.copy2(cadzow_archive, cadzow_scratch)
        print(f'{pid[:8]} Cadzow: seeded from ceph archive', flush=True)

    # Bad-channel labels precomputed by detect_bad_channels.py.  detect_bad_channels_cbin
    # only runs inside compress_bin_to_h5 when there is no Cadzow checkpoint, so on a
    # checkpoint-seeded rerun the labels would otherwise be lost.  Feeding the saved array
    # in makes the `labels` attr appear in every archive, checkpoint or not.
    labels_file = out_dir.joinpath('channel_labels.npy')
    channel_labels = np.load(labels_file) if labels_file.exists() else None
    if channel_labels is None:
        print(f'{pid[:8]} channel_labels.npy missing — run detect_bad_channels.py first', flush=True)

    out_dir.joinpath(f'{pid}_compress.error').unlink(missing_ok=True)
    try:
        one = ONE()
        ssl = SpikeSortingLoader(one=one, pid=pid)
        sr  = ssl.raw_electrophysiology(band='lf', stream=False)

        for lbl, params in PARAMS.items():
            if files[lbl].exists():
                print(f'{pid[:8]} {lbl}: exists, skipping', flush=True)
                continue
            h5tmp = files[lbl].with_suffix('.h5tmp')
            print(f'{pid[:8]} {lbl}: compressing …', flush=True)
            t0 = time.perf_counter()
            compress_bin_to_h5(
                sr.file_bin, h5tmp,
                q=Q,
                cadzow_checkpoint_file=cadzow_scratch,
                cadzow_kwargs=CADZOW_KWARGS,
                channel_labels=channel_labels,
                n_jobs=N_INNER,
                **params,
            )
            _fix_muted_attr(h5tmp)
            h5tmp.rename(files[lbl])  # atomic: sentinel only appears on success
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
parser.add_argument('--pids', nargs='*', default=None, help='explicit PID list (overrides the parquet selection; for validation runs)')
parser.add_argument('--limit', type=int, default=None, help='process at most N PIDs after array slicing (for validation runs)')
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
print(f'Task {task_id}/{n_tasks}: queuing {len(pids)} PIDs  ({N_OUTER} outer × {N_INNER} inner cores)', flush=True)

jobs = [joblib.delayed(compress_pid)(pid=pid, overwrite=args.overwrite) for pid in pids]
joblib.Parallel(n_jobs=N_OUTER, backend='loky', initializer=worker_init)(jobs)
