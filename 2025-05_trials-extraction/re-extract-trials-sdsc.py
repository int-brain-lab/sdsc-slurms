import argparse
import pickle
import logging
import shutil
import time
from pathlib import Path
from itertools import groupby
from functools import partial
from one.alf.path import ALFPath
from ibllib.oneibl.patcher import SDSCPatcher
import ibllib.pipes.dynamic_pipeline as dyn
from deploy.iblsdsc import OneSdsc, CACHE_DIR, CACHE_DIR_FI
from ibllib import __version__
from ibllib.qc.task_metrics import update_dataset_qc, TaskQC
from iblutil.io.params import FileLock
from trials_extraction.constants import (
    REVISION_FPGA, REVISION_BPOD, ROOT as POPEYE_ROOT, DATASETS, correct_version,
    PROCESSED, PROCESSED_PATHS, TASKS_DIR
)

from multiprocessing import Pool, Manager

_logger = logging.getLogger('ibllib')

assert correct_version(__version__)

N_JOBS = 8  # Number of processes to use for multiprocessing
ROOT = Path('/mnt/ibl')
PROCESSED = ROOT / PROCESSED.relative_to(POPEYE_ROOT)
TASKS_DIR = PROCESSED.with_name(TASKS_DIR.name) 
PROCESSED_PATHS = PROCESSED.with_name(PROCESSED_PATHS.name)
RUN_LIST = PROCESSED.with_stem(f'{REVISION_FPGA}_run_list')
assert RUN_LIST.exists(), f'Run list file {RUN_LIST} does not exist. Please run 02_get-trials-run-list.py first.'


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


# eids = to_process[:, 0]
# dsets = one.alyx.rest('datasets', 'list', django=f'name__startswith,_ibl_trials,session__in,{eids}')
# [{'session': url[-33:]}]
def process_one_session(tup, processed=None, processed_paths=None, flag_path=None):
    eid, session_path = tup
    session_path = ROOT.joinpath(session_path.relative_to(POPEYE_ROOT))
    # Add key to dict in case function doesn't return
    processed[eid] = (Exception('Not processed'),)
    processed_paths[eid] = []
    _logger.info('===== eID %s; %s =====', str(eid), session_path.relative_to(ROOT))
    flag = flag_path.joinpath(session_path.relative_to(ROOT))
    if flag.exists():
        _logger.critical('Skipping %s; incomplete process', session_path.relative_to(ROOT))
        return
    else:
        flag.touch()
    print(tup)

    one = OneSdsc(
        base_url='https://alyx.internationalbrainlab.org', cache_rest=None, mode='remote', cache_dir=CACHE_DIR_FI)
    sdsc_patcher = SDSCPatcher(one)
    # logic for if the dataset already exists
    # dsets = one.alyx.rest('datasets', 'list', session=eid, django='name__startswith,_ibl_trials')
    # dsets = [d for d in dsets if d['name'] in DATASETS]
    # skip = any(dsets)
    # for name, dd in groupby(dsets, lambda d: f'{d["collection"]}/{d["name"]}'):
    #     skip &= any(
    #         correct_version(d['version'])
    #         if d['version']
    #         else
    #         datetime.fromisoformat(d['created_datetime']) > datetime(2024, 7, 10)
    #         for d in dd
    #     )
    # if skip:
    #     err = (ValueError('Already run'),)
    #     return err

    try:
        tasks = list(filter(dyn.is_active_trials_task, dyn.get_trials_tasks(session_path, one=one)))
    except Exception as ex:
        processed[eid] = (ex,)
        return

    err = []
    for task in tasks:
        task.location = task.machine = 'sdsc'
        try:
            status = task.run()
            assert status == 0, 'extraction failure'
            task.outputs = [f for f in task.outputs if f.name in DATASETS]
            new_paths = task.outputs
            try:
                responses = task.register_datasets(default=True, force=False)
            except FileExistsError as e:
                if 'Protected datasets were found in the file list.' in str(e):
                    _logger.warning('Protected; forcing revision')
                    revision = determine_revision(session_path)
                    new_paths = move_to_revision_dir(task.outputs, revision)
                    if new_paths is None:
                        continue  # Permissions error
                    responses = sdsc_patcher.patch_datasets(new_paths, dry=False, versions=__version__, revision=str(revision), force=True)
                else:
                    _logger.error(e)
                    continue

            processed_paths[eid] += new_paths
            _logger.info(f'Updating dataset QC for {eid}')
            extended_qc = one.get_details(eid, True)['extended_qc']
            task_name = session_path.parts[-6]
            # if re.match(r'^Trials_\w+_\d{2}', task_name):
            #     protocol_number = int(task_name.split('_')[-1])
            # else:
            #     protocol_number = 0
            # namespace = f'task_{protocol_number:02d}'
            collection = ALFPath(new_paths[0]).collection
            if collection == 'alf':
                namespace = 'task'
            elif collection.startswith('alf/task_'):
                namespace = new_paths[0].parent.name
                if not any(k.startswith(f'_{namespace}_') for k in extended_qc):
                    # If the namespace is not present, try the default
                    namespace = 'task'
            else:
                _logger.error(f'Unexpected collection {collection} for {eid}')
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

            err.append(None)
        except Exception as ex:
            err.append(str(ex))
            print(("error!!!!!!", err, task.log),)
        finally:
            try:
                task.cleanUp()  # the popeye handler does nothing here
            except Exception as e:
                _logger.error(f'Error cleaning up task {task.name} for {eid}: {e}')
    processed[eid] = err
    flag.unlink()


def group_by_subject(run_list):
    """Group the run list by subject.
    
    Returns
    -------
    str
        The subject name.
    list of tuples
        A list of tuples where each tuple contains the eID and the session path.
    """
    # Ensure ALFPath
    run_list = [(eid, ALFPath(session_path)) for eid, session_path in run_list]
    return groupby(run_list, key=lambda x: x[1].subject)


def process_one_subject(subject, run_list, **kwargs):
    """Process all sessions for a given subject."""
    _logger.info(f'Processing subject: {subject}')
    out = {}
    for tup in run_list:
        out.update(process_one_session(tup, **kwargs))
    return {subject: out}


def save_processed(processed, processed_paths):
    """Save processed results and paths to disk."""
    with FileLock(PROCESSED, timeout_action='raise'):
        if PROCESSED.exists():
            with open(PROCESSED, 'rb') as fp:
                _processed = pickle.load(fp)
        else:
            _processed = {}
        # Update the processed dictionary with new results
        _processed.update(processed)
        with open(PROCESSED, 'wb') as fp:
            pickle.dump(_processed, fp)

    with FileLock(PROCESSED_PATHS, timeout_action='raise'):
        if PROCESSED_PATHS.exists():
            with open(PROCESSED_PATHS, 'rb') as fp:
                _processed_paths = pickle.load(fp)
        else:
            _processed_paths = {}
        # Update the processed paths dictionary with new results
        _processed_paths.update(processed_paths)
        with open(PROCESSED_PATHS, 'wb') as fp:
            pickle.dump(_processed_paths, fp)


def main():
    parser = argparse.ArgumentParser(description='Re-extract trials from Popeye sessions.')
    parser.add_argument('--n-sessions', type=int, default=2,
                        help='Number of sessions to process (default: 2)')
    parser.add_argument('--n-subjects', type=int)
    parser.add_argument('--n-jobs', type=int, default=N_JOBS)
    parsed = parser.parse_args()

    with FileLock(RUN_LIST, timeout_action='raise'):
        with open(RUN_LIST, 'rb') as fp:
            run_list = pickle.load(fp)
    if PROCESSED.exists():
        with FileLock(PROCESSED, timeout_action='raise'):
            with open(PROCESSED, 'rb') as fp:
                _processed = pickle.load(fp)
            l_before= len(run_list)
            run_list = list(filter(lambda x: x[0] not in _processed, run_list))
            _logger.info(f'Skipping {l_before - len(run_list)} sessions processed but not registered.')

    try:
        flag = Path(__file__).resolve().with_name('processing')
    except NameError:
        flag = Path.home().joinpath('Documents', 'PYTHON', 'sdsc-slurms', '2025-05_trials-extraction', 'processing')
    flag.mkdir(exist_ok=True)

    manager = Manager()
    # Initialize shared dictionaries for processed results and paths
    processed = manager.dict()  # Store processed results
    processed_paths = manager.dict()  # Store processed paths

    if parsed.n_subjects is not None:
        n = parsed.n_subjects
        to_run = []
        for subject, run_list_ in group_by_subject(run_list):
            if len(to_run) < n:
                to_run.append((subject, list(run_list_)))
            else:
                break
        f = partial(process_one_subject, processed=processed, processed_paths=processed_paths, flag_path=flag)
    else:
        n = parsed.n_sessions
        to_run = run_list[:n]
        f = partial(process_one_session, processed=processed, processed_paths=processed_paths, flag_path=flag)

    n_jobs = parsed.n_jobs
    t0 = time.time()
    _logger.info('Starting processing of %i sessions with %i jobs...', len(to_run), n_jobs)
    with Pool(processes=n_jobs) as pool:
        results = pool.map(f, to_run)
    _logger.info(f'Processing %i sessions with %i jobs took %.2g minutes', len(to_run), n_jobs, (time.time() - t0) / 60)
    # Save processed paths
    _logger.info('Saving processed results and paths...')
    save_processed(dict(processed), dict(processed_paths))
    # cmd = f'chmod -R 777 {str(TASKS_DIR)}/Trials*'
    # os.system(cmd)
    # for p in (PROCESSED_PATHS, PROCESSED):
    #     cmd = f'chmod -R 777 {str(p)}'
    #     os.system(cmd)


if __name__ == '__main__':
    main()



