import os
import logging
import pickle
import shutil
from pathlib import Path
from itertools import groupby

from ibllib import __version__
from ibllib.oneibl.patcher import SDSCPatcher
from ibllib.qc.task_metrics import update_dataset_qc, TaskQC

from iblutil.util import rrmdir
from iblutil.io.params import FileLock

from one.alf.path import get_session_path

from deploy.iblsdsc import OneSdsc, CACHE_DIR_FI

from trials_extraction.constants import DATASETS, correct_version, ROOT as POPEYE_ROOT, PROCESSED, PROCESSED_PATHS, REVISION_FPGA, REVISION_BPOD

assert correct_version(__version__)
_logger = logging.getLogger('ibllib')
ROOT = Path('/mnt/ibl')
PROCESSED = ROOT / PROCESSED.relative_to(POPEYE_ROOT)
PROCESSED_PATHS = PROCESSED.with_name(PROCESSED_PATHS.name)

assert PROCESSED.exists(), f'Processed file {PROCESSED} does not exist'
assert PROCESSED_PATHS.exists(), f'Processed paths file {PROCESSED_PATHS} does not exist'
assert os.access(PROCESSED, os.W_OK), f'Processed file {PROCESSED} is not writable'
assert os.access(PROCESSED_PATHS, os.W_OK), f'Processed paths file {PROCESSED_PATHS} is not writable'


def move_to_revision_dir(paths, revision=REVISION_FPGA):
    base_dir = paths[0].parent
    # A lot appear to have already been moved so if the folder already exists,
    # simply return those paths
    rev_dir = base_dir.joinpath(f'#{revision}#')
    new_paths = [rev_dir.joinpath(p.name) for p in paths]
    if rev_dir.exists():
        if all(map(Path.exists, new_paths)):
            return new_paths
        else:  # just start again (this may lead to file not found error)
            shutil.rmtree(rev_dir)
    try:
        rev_dir.mkdir()
    except PermissionError:
        _logger.error('Couldn\'t register:')
        _logger.error(paths)
        _logger.error('Add datauser permissions on Popeye')
        return
    new_paths = []
    for p in paths:
        new_paths.append(p.rename(rev_dir.joinpath(p.name)))
    return new_paths


def determine_revision(session_path):
    """Determine the revision based on the session path."""
    session_path = ROOT.joinpath(*session_path.parts[-5:])
    daq_dirs = (session_path.joinpath(d).exists() for d in ['raw_sync_data', 'raw_ephys_data'])
    return REVISION_FPGA if any(daq_dirs) else REVISION_BPOD


def main():
    # Load the processed paths
    with FileLock(PROCESSED_PATHS, timeout_action='raise'):
        with open(PROCESSED_PATHS, 'rb') as fp:
            processed_paths = pickle.load(fp)
    with FileLock(PROCESSED, timeout_action='raise'):
        with open(PROCESSED, 'rb') as fp:
            processed = pickle.load(fp)

    # one = ONE(cache_rest=None)
    one = OneSdsc(cache_rest=None, mode='remote', cache_dir=CACHE_DIR_FI)
    sdsc_patcher = SDSCPatcher(one)

    to_remove_processed = []
    to_remove_paths = {}
    # Begin registration
    for eid, paths in processed_paths.items():
        if not (all(map(Path.exists, paths)) and set(p.name for p in paths) >= set(DATASETS)):
            _logger.info(f'Re-extract required: {eid}, {paths[0].parents[1]}')
            to_remove_paths[eid] = paths
            if eid in processed and not any(processed[eid]):
                to_remove_processed.append(eid)  # removed from processed so it will be re-extracted
            continue

        # Group by task here
        for session_path, pp in groupby(paths, lambda x: get_session_path(x)):
            pp = list(pp)
            responses = []
            try:
                responses = sdsc_patcher.patch_datasets(pp, dry=False, versions=__version__)
            except FileExistsError as e:
                if 'Protected datasets were found in the file list.' in str(e):
                    _logger.warning('Protected; forcing revision')
                    revision = determine_revision(session_path)
                    new_paths = move_to_revision_dir(pp, revision)
                    if new_paths is None:
                        continue  # Permissions error
                    responses = sdsc_patcher.patch_datasets(new_paths, dry=False, versions=__version__, revision=str(revision), force=True)
                else:
                    _logger.error(e)
                    continue
            except Exception as e:
                _logger.error(f'Error | {e}')
                continue
            
            # Add paths to remove list
            if eid not in to_remove_paths:
                to_remove_paths[eid] = pp
            else:
                to_remove_paths[eid] += pp
            # Clean up the paths
            for p in pp:
                try:
                    p.unlink()  # remove the symlink
                except OSError as e:
                    _logger.error(f'Error unlinking {p}: {e}')
            # Attempt to remove empty directories
            # Removes up to task name dir (we don't have the permissions)
            rrmdir(pp[0].parent, levels=5)
            
            # Update the dataset QC
            _logger.info(f'Updating dataset QC for {eid}')
            extended_qc = one.get_details(eid, True)['extended_qc']
            task_name = session_path.parts[-6]
            # if re.match(r'^Trials_\w+_\d{2}', task_name):
            #     protocol_number = int(task_name.split('_')[-1])
            # else:
            #     protocol_number = 0
            # namespace = f'task_{protocol_number:02d}'
            if pp[0].parent.name == 'alf':
                namespace = 'task'
            elif pp[0].parent.name.startswith('task_'):
                namespace = pp[0].parent.name
                if not any(k.startswith(f'_{namespace}_') for k in extended_qc):
                    # If the namespace is not present, try the default
                    namespace = 'task'
            else:
                _logger.error(f'Unexpected parent name {pp[0].parent.name} for {eid}')
                continue
            
            results = {k: v for k, v in extended_qc.items() if k.startswith(f'_{namespace}_')}
            if not results:
                _logger.warning(f'No QC found for {task_name} in {eid}')
                continue
            try:
                qc = TaskQC(session_path, namespace=namespace, one=one)
                qc.passed = results
                responses = update_dataset_qc(qc, responses, one)
            except Exception as e:
                _logger.error(f'Error updating dataset QC for {eid}: {e}')
                continue
        if eid in processed and not any(processed[eid]):
            to_remove_processed.append(eid)  # removed from processed so it will be re-extracted

    with FileLock(PROCESSED_PATHS, timeout_action='raise'):
        with open(PROCESSED_PATHS, 'rb') as fp:
            processed_paths_ = pickle.load(fp)
        for eid, paths in to_remove_paths.items():
            if eid in processed_paths_:
                processed_paths_[eid] = [p for p in processed_paths_[eid] if p not in paths]
                if not processed_paths_[eid]:
                    del processed_paths_[eid]  # remove empty entries
        with open(PROCESSED_PATHS, 'wb') as fp:
            pickle.dump(processed_paths_, fp)
        
    with FileLock(PROCESSED, timeout_action='raise'):
        with open(PROCESSED, 'rb') as fp:
            processed_ = pickle.load(fp)
        for eid in to_remove_processed:
            if eid in processed_:
                del processed_[eid]
        with open(PROCESSED, 'wb') as fp:
            pickle.dump(processed_, fp)
    
    RUN_LIST = PROCESSED.with_name('2025-03-03_run_list.pkl')
    with FileLock(RUN_LIST, timeout_action='raise'):
        with open(RUN_LIST, 'rb') as fp:
            run_list = pickle.load(fp)
        # Remove entries for processed sessions
        run_list = [tup for tup in run_list if tup[0] not in to_remove_processed]
        with open(RUN_LIST, 'wb') as fp:
            pickle.dump(run_list, fp)
