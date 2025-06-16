import argparse
import pickle
from datetime import timedelta
from itertools import groupby
from one.alf.path import ALFPath
from ibllib import __version__
from iblutil.io.params import FileLock
from trials_extraction.constants import REVISION_FPGA, PROCESSED


N_JOBS = 48  # Number of processes to use for multiprocessing
AVG_LEN = 40.  # Mean extraction time per session in seconds
RUN_LIST = PROCESSED.with_stem(f'{REVISION_FPGA}_run_list')
assert RUN_LIST.exists(), f'Run list file {RUN_LIST} does not exist. Please run 02_get-trials-run-list.py first.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-extract trials from Popeye sessions.')
    parser.add_argument('--n-sessions', type=int, default=2,
                        help='Number of sessions to process (default: 2)')
    parser.add_argument('--n-subjects', type=int)
    parsed = parser.parse_args()

    with FileLock(RUN_LIST, timeout_action='raise'):
        with open(RUN_LIST, 'rb') as fp:
            run_list = pickle.load(fp)

    if parsed.n_subjects is not None:
        n = parsed.n_subjects
        n_sessions = 0
        run_list = [(eid, ALFPath(session_path)) for eid, session_path in run_list]
        run_list = groupby(run_list, key=lambda x: x[1].session_parts[1])
        for i, (_, run_list_) in enumerate(run_list):
            if i < n:
                n_sessions += len(list(run_list_))
            else:
                break
    else:
        n_sessions = min(parsed.n_sessions, len(run_list))
        n = len(set(ALFPath(session_path).session_parts[1] for _, session_path in run_list[:n_sessions]))

    estimate = timedelta(seconds=(n_sessions * AVG_LEN) / N_JOBS)
    # Format the estimate as a string with format 'HH:MM:SS'
    estimate_str = str(estimate).split('.')[0]  # Remove microseconds
    print(f'Processing {n_sessions} sessions for {n} subjects (should take about {estimate_str})')
    # TODO update batch with args and estimate