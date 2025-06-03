import os
import pickle
import logging
from itertools import groupby
from functools import partial
from one.alf.path import ALFPath
import ibllib.pipes.dynamic_pipeline as dyn
from deploy.iblsdsc import OneSdsc, CACHE_DIR, CACHE_DIR_FI
from ibllib import __version__
from iblutil.io.params import FileLock
from trials_extraction.constants import REVISION_FPGA, ROOT, DATASETS, correct_version, PROCESSED, PROCESSED_PATHS, TASKS_DIR, setenv

from multiprocessing import Pool, Manager

logger = logging.getLogger('ibllib')

assert correct_version(__version__)
setenv()

RUN_LIST = PROCESSED.with_stem(f'{REVISION_FPGA}_run_list')
assert RUN_LIST.exists(), f'Run list file {RUN_LIST} does not exist. Please run 02_get-trials-run-list.py first.'

def popeye_to_sdsc(path):
    return CACHE_DIR_FI.joinpath(path.relative_to(CACHE_DIR))

# eids = to_process[:, 0]
# dsets = one.alyx.rest('datasets', 'list', django=f'name__startswith,_ibl_trials,session__in,{eids}')
# [{'session': url[-33:]}]
def process_one_session(tup, processed=None, processed_paths=None):
    eid, session_path = tup
    # Add key to dict in case function doesn't return
    processed[eid] = (Exception('Not processed'),)
    processed_paths[eid] = []
    logger.info('===== eID %s; %s =====', str(eid), session_path.relative_to(ROOT))
    print(tup)

    one = OneSdsc(
        base_url='https://alyx.internationalbrainlab.org', cache_rest=None, mode='remote', cache_dir=CACHE_DIR_FI)

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
        tasks = dyn.get_trials_tasks(session_path, one=one)
    except Exception as ex:
        processed[eid] = (ex,)
        return

    err = []
    for task in filter(dyn.is_active_trials_task, tasks):
        task.location = task.machine = 'popeye'
        try:
            status = task.run()
            assert status == 0, 'extraction failure'
            processed_paths[eid] += [popeye_to_sdsc(f) for f in task.outputs if f.name in DATASETS]
        except Exception as ex:
            err.append(str(ex))
            print(("error!!!!!!", err, task.log),)
        finally:
            try:
                task.cleanUp()  # remove symlinks, etc.
                # To keep number of files low, remove any other extracted datasets
                for f in task.outputs:
                    if not f.name in DATASETS:
                        f.unlink()
            except AssertionError:
                pass
    processed[eid] = err


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
    return groupby(run_list, key=lambda x: x[1].session_parts[1])


def process_one_subject(subject, run_list):
    """Process all sessions for a given subject."""
    logger.info(f'Processing subject: {subject}')
    out = {}
    for tup in run_list:
        out.update(process_one_session(tup))
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
    with FileLock(RUN_LIST, timeout_action='raise'):
        with open(RUN_LIST, 'rb') as fp:
            run_list = pickle.load(fp)

    manager = Manager()
    # Initialize shared dictionaries for processed results and paths
    processed = manager.dict()  # Store processed results
    processed_paths = manager.dict()  # Store processed paths

    f = partial(process_one_session, processed=processed, processed_paths=processed_paths)
    with Pool(processes=2) as pool:
        results = pool.map(f, run_list[:2])
    assert len(processed) == 2
    assert len(processed_paths) == 2
    # Save processed paths
    save_processed(dict(processed), dict(processed_paths))
    cmd = f'chmod -R 777 {str(TASKS_DIR)}/Trials*'
    os.system(cmd)


if __name__ == '__main__':
    main()



