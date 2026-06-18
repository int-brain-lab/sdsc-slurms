"""Single-PID smoke test — run interactively on Popeye to validate the pipeline before sbatch."""
import os
import time
from pathlib import Path

from deploy.iblsdsc import OneSdsc as ONE
from brainbox.io.one import SpikeSortingLoader
from lfpack import compress_bin_to_h5

os.environ['TQDM_DISABLE'] = '0'

pid = '1a276285-8b0e-4cc9-9f0a-a3a002978724'  # benchmark PID

Q = 10
N_JOBS = 12
PARAMS = {
    'default':    dict(epsilon=150.0, alpha=28.0),
    'aggressive': dict(epsilon=450.0, alpha=96.0),
}
CADZOW_KWARGS = dict(rank=5, niter=1, fmax=None, nswx=64, gap_threshold=2.0, ppca_k=2.0)

SCRATCH_ROOT = Path(os.environ.get('SCRATCH_ROOT', '/scratch/lfpack_debug'))
OUTPUT_ROOT  = Path('/mnt/home/owinter/ceph/lfpack')

scratch_dir = SCRATCH_ROOT.joinpath(pid)
scratch_dir.mkdir(parents=True, exist_ok=True)
cadzow_file = scratch_dir.joinpath('lf_resampled_car_cadzow.npy')

out_dir = OUTPUT_ROOT.joinpath(pid)
out_dir.mkdir(parents=True, exist_ok=True)

one = ONE()
ssl = SpikeSortingLoader(one=one, pid=pid)
sr  = ssl.raw_electrophysiology(band='lf', stream=False)

for lbl, params in PARAMS.items():
    out_h5 = out_dir.joinpath(f'lf_compressed{"_" + lbl if lbl != "default" else ""}.h5')
    print(f'{pid[:8]} {lbl}: compressing …', flush=True)
    t0 = time.perf_counter()
    compress_bin_to_h5(
        sr.file_bin, out_h5,
        q=Q,
        cadzow_checkpoint_file=cadzow_file,
        cadzow_kwargs=CADZOW_KWARGS,
        n_jobs=N_JOBS,
        **params,
    )
    print(f'{pid[:8]} {lbl}: done in {time.perf_counter() - t0:.1f} s')

print('Scratch checkpoint size:', cadzow_file.stat().st_size / 1e9, 'GB')
