"""
Reduce step: aggregate per-PID lf_compressed*.h5 files into a pair of
multi-recording HDF5 archives.

Scans OUTPUT_ROOT for subdirectories that contain the Cadzow checkpoint
(lf_resampled_car_cadzow.npy), which is the completion sentinel for the
expensive decimation stage.  For each such directory it also checks that
the corresponding lf_compressed.h5 / lf_compressed_aggressive.h5 exist
before adding them to the combined file.

Output files (written to OUTPUT_ROOT):
    lf_compressed_all.h5                default parameters  (ε=150, α=28)
    lf_compressed_aggressive_all.h5     aggressive params   (ε=450, α=96)
    lf_compressed_all_bwm.h5            default, BWM only   (--bwm)
    lf_compressed_aggressive_all_bwm.h5 aggressive, BWM only (--bwm)

Each per-PID file stores the recording under its original binary-stem key;
this script supplies a recording_map so each group is stored under the PID
UUID in the merged file.

With --bwm the output is restricted to the Brain-Wide Map freeze insertions
(df_probes['bwm'] == True).  Missing BWM PIDs (not yet computed) are reported
as warnings so completeness can be tracked.

Usage
-----
# Aggregate all ready PIDs (both compression passes):
python reduce.py

# Aggregate only BWM freeze PIDs and report missing ones:
python reduce.py --bwm

# Aggregate BWM PIDs, default pass only:
python reduce.py --bwm --passes default

# Custom output root:
python reduce.py --output-root /path/to/cells --bwm
"""
import argparse
import logging
from pathlib import Path

import ephysatlas.data
from lfpack import merge_h5

# ── paths ──────────────────────────────────────────────────────────────────────
OUTPUT_ROOT  = Path('/mnt/home/owinter/ceph/ea/denoised_lfp')
PROJECT_ROOT = Path.home().joinpath('data', 'ephys-atlas', 'projects')
PROJECT      = 'ibl_neuropixel_brainwide_01'
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


def load_bwm_pids(project_root: Path, project: str) -> list[str]:
    """Return sorted list of BWM freeze PIDs from the probe details parquet.

    Downloads probe details from S3 if not cached locally.

    Parameters
    ----------
    project_root : Path
        Root directory containing one sub-folder per project.
    project : str
        Project name (e.g. ``'ibl_neuropixel_brainwide_01'``).

    Returns
    -------
    list[str]
        Sorted PID strings for BWM freeze insertions.
    """
    path_project = project_root.joinpath(project)
    if not path_project.joinpath('df_probe_details.pqt').exists():
        from one.api import ONE
        one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
        ephysatlas.data.download_probe_details(project_root, project=project, one=one)
    df_probes = ephysatlas.data.read_probe_details(path_project)
    bwm_pids = sorted(df_probes.loc[df_probes['bwm'], 'pid'].tolist())
    log.info(f'BWM freeze: {len(bwm_pids)} PIDs in project "{project}"')
    return bwm_pids


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
    parser.add_argument(
        '--bwm', action='store_true',
        help='Restrict to Brain-Wide Map freeze PIDs only and report missing ones.',
    )
    parser.add_argument(
        '--project-root', type=Path, default=PROJECT_ROOT,
        help='Root directory for ephys-atlas project parquet files.',
    )
    parser.add_argument(
        '--project', default=PROJECT,
        help='Ephys-atlas project name (used with --bwm).',
    )
    args = parser.parse_args()

    pids = find_ready_pids(args.output_root)

    if args.bwm:
        bwm_pids = load_bwm_pids(args.project_root, args.project)
        bwm_set  = set(bwm_pids)
        ready_set = set(pids)

        missing = sorted(bwm_set - ready_set)
        if missing:
            log.warning(f'{len(missing)} BWM PID(s) not yet computed (no Cadzow checkpoint):')
            for pid in missing:
                log.warning(f'  missing: {pid}')
        else:
            log.info('All BWM PIDs have a Cadzow checkpoint — dataset is complete.')

        pids = sorted(bwm_set & ready_set)
        log.info(f'Restricting to {len(pids)} BWM PID(s) with Cadzow checkpoint')
    log.info(f'PIDs: {pids[:5]}{"…" if len(pids) > 5 else ""}')

    for pass_name in args.passes:
        src_name, dst_name = PASSES[pass_name]
        if args.bwm:
            dst_name = dst_name.replace('.h5', '_bwm.h5')
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