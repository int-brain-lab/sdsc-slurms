"""
Reduce step: aggregate per-PID lf_compressed*.h5 files into a pair of
multi-recording HDF5 archives.

Scans OUTPUT_ROOT for subdirectories that contain the Cadzow checkpoint
(lf_resampled_car_cadzow.npy), which is the completion sentinel for the
expensive decimation stage.  For each such directory it also checks that
the corresponding lf_compressed.h5 / lf_compressed_aggressive.h5 exist
before adding them to the combined file.

Output files (written to OUTPUT_ROOT):
    lf_compressed_all.h5            default parameters  (ε=150, α=28)
    lf_compressed_aggressive_all.h5 aggressive params   (ε=450, α=96)

Each per-PID file stores the recording under its original binary-stem key;
this script supplies a recording_map so each group is stored under the PID
UUID in the merged file.
"""
import argparse
import logging
from pathlib import Path

from lfpack import merge_h5

# ── paths ──────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path('/mnt/home/owinter/ceph/ea/cells')
SENTINEL     = 'lf_resampled_car_cadzow.npy'

PASSES = {
    'default':    ('lf_compressed.h5',            'lf_compressed_all.h5'),
    'aggressive': ('lf_compressed_aggressive.h5', 'lf_compressed_aggressive_all.h5'),
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def find_ready_pids(output_root: Path) -> list[str]:
    """Return sorted list of PIDs whose Cadzow checkpoint exists.

    Parameters
    ----------
    output_root : Path
        Root directory containing one sub-folder per PID.

    Returns
    -------
    list[str]
        Sorted PID strings.
    """
    pids = sorted(
        p.name
        for p in output_root.iterdir()
        if p.is_dir() and p.joinpath(SENTINEL).exists()
    )
    log.info(f'Found {len(pids)} PID(s) with Cadzow checkpoint under {output_root}')
    return pids


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate per-PID lfpack H5 files.')
    parser.add_argument(
        '--output-root', type=Path, default=OUTPUT_ROOT,
        help='Root directory containing one sub-folder per PID.',
    )
    parser.add_argument(
        '--passes', nargs='+', choices=list(PASSES), default=list(PASSES),
        help='Which compression pass(es) to aggregate (default: both).',
    )
    args = parser.parse_args()

    pids = find_ready_pids(args.output_root)
    log.info(f'PIDs: {pids[:5]}{"…" if len(pids) > 5 else ""}')

    for pass_name in args.passes:
        src_name, dst_name = PASSES[pass_name]
        src_files = [args.output_root.joinpath(pid, src_name) for pid in pids]
        # Keep only PIDs whose H5 is already written; others are still compressing.
        ready = [(pid, p) for pid, p in zip(pids, src_files) if p.exists()]
        n_missing = len(pids) - len(ready)
        if n_missing:
            log.info(f'{pass_name}: {n_missing} PID(s) not yet ready, skipping')

        dst_path = args.output_root.joinpath(dst_name)
        recording_map = {p: pid for pid, p in ready}
        log.info(f'{pass_name}: merging {len(ready)} recordings → {dst_path.name}')
        merge_h5([p for _, p in ready], dst_path, recording_map=recording_map)
        log.info(f'{pass_name}: done')

    log.info('Done.')


if __name__ == '__main__':
    main()