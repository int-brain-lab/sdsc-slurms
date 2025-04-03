import argparse
import joblib
from pathlib import Path

import numpy as np
from deploy.iblsdsc import OneSdsc as ONE

from brainwidemap.bwm_loading import bwm_query
from prior_localization.fit_data import fit_session_ephys

n_jobs = 48

N_PSEUDO = 100
N_PER_JOB = 4
OUTPUT_DIR = '/mnt/home/owinter/ceph/bwm/wheel_rerun'
TARGET = 'wheel-velocity'

WORKER_ID, N_WORKERS = (2, 3)  # 0-based worker id
job_list = np.arange(1, 3000 + 1)[WORKER_ID::N_WORKERS]


# job_list = np.load('/mnt/home/owinter/Documents/sdsc-slurms/bwm/jobs_remaining.npy')


def run_single_job(job_idx):

    output_dir = Path(OUTPUT_DIR).joinpath(TARGET)

    # Get session idx
    session_idx = int(np.ceil(job_idx / (N_PSEUDO / N_PER_JOB)) - 1)

    # Set of pseudo sessions for one session
    all_pseudo = list(range(N_PSEUDO))
    # Select relevant pseudo sessions for this job
    pseudo_idx = int((job_idx - 1) % (N_PSEUDO / N_PER_JOB) * N_PER_JOB)
    pseudo_ids = all_pseudo[pseudo_idx:pseudo_idx + N_PER_JOB]
    # Shift by 1; old array starts at 0, pseudo ids should start at 1
    pseudo_ids = list(np.array(pseudo_ids) + 1)
    # Add real session to first pseudo block
    if pseudo_idx == 0:
        pseudo_ids = [-1] + pseudo_ids

    # Create an offline ONE instance, we don't want to hit the database when running so many jobs in parallel and have
    # downloaded the data before
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')

    # Get info for respective eid from bwm_dataframe
    bwm_df = bwm_query(one=one, freeze='2023_12_bwm_release')
    session_id = bwm_df.eid.unique()[session_idx]
    subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
    # We are merging probes per session, therefore using a list of all probe names of a session as input
    probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)
    probe_name = probe_name[0] if len(probe_name) == 1 else probe_name

    # set BWM defaults here
    binsize = None
    n_bins_lag = None
    n_bins = None
    n_runs = 10

    if TARGET == 'stimside':
        align_event = 'stimOn_times'
        time_window = (0.0, 0.1)
        saturation_intervals = 'saturation_stim_plus01'
        model = 'oracle'
        estimator = 'LogisticRegression'

    elif TARGET == 'signcont':
        align_event = 'stimOn_times'
        time_window = (0.0, 0.1)
        saturation_intervals = 'saturation_stim_plus01'
        model = 'oracle'
        estimator = 'Lasso'

    elif TARGET == 'choice':
        align_event = 'firstMovement_times'
        time_window = (-0.1, 0.0)
        saturation_intervals = 'saturation_move_minus02'
        model = 'actKernel'
        estimator = 'LogisticRegression'

    elif TARGET == 'feedback':
        align_event = 'feedback_times'
        time_window = (0.0, 0.2)
        saturation_intervals = 'saturation_feedback_plus04'
        model = 'actKernel'
        estimator = 'LogisticRegression'

    elif TARGET == 'pLeft':
        align_event = 'stimOn_times'
        time_window = (-0.6, -0.1)
        saturation_intervals = 'saturation_stim_minus06_plus06'
        model = 'optBay'
        estimator = 'LogisticRegression'

    elif TARGET in ['wheel-speed', 'wheel-velocity']:
        align_event = 'firstMovement_times'
        time_window = (-0.2, 1.0)
        saturation_intervals = 'saturation_move_minus02'
        model = 'oracle'
        estimator = 'Lasso'
        binsize = 0.02
        n_bins_lag = 10
        n_bins = 60
        n_runs = 2
    else:
        raise ValueError(f'{TARGET} is an invalid target value')

    # Run the decoding for the current set of pseudo ids.
    results = fit_session_ephys(
        one, session_id, subject, probe_name, output_dir=output_dir, pseudo_ids=pseudo_ids, target=TARGET,
        align_event=align_event, time_window=time_window,
        saturation_intervals=saturation_intervals,
        model=model, n_runs=n_runs,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        compute_neurometrics=False, motor_residuals=False,
    )
    # Print out success string so we can easily sweep through error logs
    print(f'Job with ID {job_idx} successful')

# %%
jobs = [joblib.delayed(run_single_job)(job_idx=jid) for jid in job_list]
joblib.Parallel(n_jobs=n_jobs)(jobs)
