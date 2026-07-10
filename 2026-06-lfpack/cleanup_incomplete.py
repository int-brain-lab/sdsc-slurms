"""
Remove .h5 and .h5tmp files from PID folders that have no cadzow .npy archive.

The cadzow archive (lf_resampled_car_cadzow.npy) is written to ceph only after
both compression passes complete successfully.  Its absence means the folder is
either mid-run or was killed before finishing — any .h5 files present may be
corrupt or partial and should be removed so the next run starts clean.

Usage
-----
    python cleanup_incomplete.py          # dry run — prints what would be deleted
    python cleanup_incomplete.py --delete # actually deletes
"""
import argparse
from pathlib import Path

OUTPUT_ROOT = Path('/mnt/home/owinter/ceph/ea/denoised_lfp')

parser = argparse.ArgumentParser()
parser.add_argument('--delete', action='store_true', help='Actually delete (default: dry run)')
args = parser.parse_args()

to_remove = []
for pid_dir in sorted(OUTPUT_ROOT.iterdir()):
    if not pid_dir.is_dir():
        continue
    if pid_dir.joinpath('lf_resampled_car_cadzow.npy').exists():
        continue  # fully completed, leave it alone
    candidates = list(pid_dir.glob('*.h5')) + list(pid_dir.glob('*.h5tmp'))
    if candidates:
        to_remove.extend(candidates)

if not to_remove:
    print('Nothing to remove.')
else:
    print(f'{"DRY RUN — " if not args.delete else ""}removing {len(to_remove)} file(s):')
    for f in to_remove:
        print(f'  {f}')
        if args.delete:
            f.unlink()
    if not args.delete:
        print('\nRe-run with --delete to actually remove.')