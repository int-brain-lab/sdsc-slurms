"""Data locations, revision names and version numbers for the trials extraction script.

--------------------
Revision: 2025-03-03
Version: 3.3.0
--------------------
This version includes the fix for incorrect first trial extraction in some FPGA sessions.
https://github.com/int-brain-lab/ibllib/issues/909
https://github.com/int-brain-lab/ibllib/pull/943 - release

--------------------
Revision: 2024-07-15
Version: 2.38.0
--------------------
This version extracts stimulus times by taking first TTL time within a fixed window after the Bpod
trigger.  At this time, the Bpod trigger times were also saved by default.
https://github.com/int-brain-lab/iblrig/issues/654
https://github.com/int-brain-lab/ibllib/issues/775
https://github.com/int-brain-lab/ibllib/pull/788 - feature branch
https://github.com/int-brain-lab/ibllib/pull/802 - release

"""
from pathlib import Path
from packaging import version

REVISION_FPGA = '2025-03-03'
REVISION_BPOD = '2024-07-15'
VERSION_FPGA = version.Version('3.3.0')
VERSION_BPOD = version.Version('2.38.0')
ROOT = Path('/mnt/sdceph/users/ibl/data')
TASKS_DIR = ROOT.joinpath('quarantine', f'{REVISION_FPGA}_trials-extraction', 'tasks')
PROCESSED = ROOT.joinpath('quarantine', f'{REVISION_FPGA}_trials-extraction', f'{REVISION_FPGA}_stim-times.pkl')
PROCESSED_PATHS = ROOT.joinpath('quarantine', f'{REVISION_FPGA}_trials-extraction', f'{REVISION_FPGA}_paths.pkl')
DATASETS = ('_ibl_trials.table.pqt', '_ibl_trials.stimOff_times.npy',
            '_ibl_trials.stimOnTrigger_times.npy', '_ibl_trials.stimOffTrigger_times.npy')

def correct_version(v, min_version=VERSION_FPGA):
    try:
        return version.parse(v or '') >= min_version
    except version.InvalidVersion:
        return False

def setenv():
    """The Popeye patcher uses this path for something"""
    import os
    if not os.getenv('SDSC_PATCH_PATH'):
        os.environ['SDSC_PATCH_PATH'] = TASKS_DIR.as_posix()
