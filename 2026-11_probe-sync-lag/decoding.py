"""BWM decoding of stimulus side, choice and feedback for the BWM sessions of `pids.csv`, with the original and with the
re-extracted spike times of the affected probe.

Reproduces the BWM paper run (prior-localization branch `sirena`, `run_scripts/run_ephys_decoding.py`): probes merged per
session, good units of spike sorting revision 2024-05-06, L1 logistic regression, 10 runs of nested 5-fold cross
validation, 200 pseudo-sessions. The only change in the `shifted` condition is the spike times of the affected probe,
remapped through the re-extracted probe sync (`sync_lag.py --step extract`):
`t_shifted = sync_new(sync_original^-1(t))`, i.e. back to probe time with the sync used for the release, then to session
time with the last-edge pairing.

Prior (`pLeft`, after the 408 BWM tasks so that their indices do not change): decoding of the Bayes-optimal prior as in
the prior paper (Findling et al. 2025): ITI window (-0.6, -0.1) s before stimulus onset, `optBay` model, L1 linear
regression (Lasso), R2 score, at least 250 trials, 200 pseudo-sessions.

One task per (pid, target, condition, block of 50 pseudo-sessions): 17 x 4 x 2 x 4 = 544 tasks, each with its own
output folder `decoding/<condition>/<target>/block_<k>` so that concurrent tasks never write the same behaviour-model
cache. Each SLURM array element takes a full node and runs its share of the tasks (`tasks[node::n_nodes]`) in
parallel processes. Tasks are idempotent: a finished task leaves `decoding/_done/<task>`, a failed one
`decoding/_errors/<task>.error`.

    python decoding.py --stage                        # login node: BWM trials table (saturation flags, needs network)
    python decoding.py --node 0 --n-nodes 3           # one array element, $SLURM_ARRAY_TASK_ID
    python decoding.py --task 0                       # a single task, for debugging
    python decoding.py --aggregate                    # per session x region scores, p-values, null medians -> decode.pqt
"""

import os

os.environ["TQDM_DISABLE"] = "1"
import argparse
import functools
import pickle
import re
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import brainwidemap.bwm_loading
import prior_localization.prepare_data
from brainwidemap.bwm_loading import bwm_query
from deploy.iblsdsc import OneSdsc as ONE
from prior_localization.fit_data import fit_session_ephys
from prior_localization.functions.utils import check_config

OUTPUT_PATH = Path("/mnt/home/owinter/ceph/sync_lag")
DECODING_PATH = OUTPUT_PATH.joinpath("decoding")
BWM_TABLES = OUTPUT_PATH.joinpath("bwm_tables")
PIDS = pd.read_csv(Path(__file__).parent.joinpath("pids.csv"))
PIDS = PIDS[PIDS.bwm].reset_index(drop=True)
TARGETS = ["stimside", "choice", "feedback"]  # BWM paper
PRIOR_TARGETS = ["pLeft"]  # prior paper, tasks appended after the BWM ones
CONDITIONS = ["original", "shifted"]
N_PSEUDO, N_PER_JOB = 200, 50
# settings per target: align event, time window (s), saturation interval, null model, estimator, minimum trials
BWM_TARGETS = {
    "stimside": ("stimOn_times", (0.0, 0.1), "saturation_stim_plus01", "oracle", "LogisticRegression", 1),
    "choice": ("firstMovement_times", (-0.1, 0.0), "saturation_move_minus02", "actKernel", "LogisticRegression", 1),
    "feedback": ("feedback_times", (0.0, 0.2), "saturation_feedback_plus04", "actKernel", "LogisticRegression", 1),
    "pLeft": ("stimOn_times", (-0.6, -0.1), "saturation_stim_minus06_plus06", "optBay", "Lasso", 250),
}
_download_aggregate_tables = brainwidemap.bwm_loading.download_aggregate_tables


def patch_aggregate_tables():
    """
    Point `brainwidemap.bwm_loading.download_aggregate_tables` to the staged BWM paper trials table.

    The trials table holding the saturation flags is downloaded from AWS once (`--stage`) into a writable folder, and
    the tag is pinned: the brainwidemap default (2026_Q2) has no trials table checksum and raises KeyError. Called in
    every process: joblib workers import brainwidemap afresh, without the patches of the main process.
    """
    brainwidemap.bwm_loading.download_aggregate_tables = functools.partial(
        _download_aggregate_tables, target_path=BWM_TABLES, tag="2024_Q2_IBL_et_al_BWM"
    )


patch_aggregate_tables()


def tasks():
    """Table of the array tasks: pid, target, condition and pseudo-session ids."""
    rows = []
    for targets in [TARGETS, PRIOR_TARGETS]:  # the prior tasks come after all the BWM tasks: stable task indices
        for pid in PIDS.pid:
            for target in targets:
                for condition in CONDITIONS:
                    for block, start in enumerate(range(0, N_PSEUDO, N_PER_JOB)):
                        pseudo_ids = list(range(start + 1, start + N_PER_JOB + 1))
                        rows.append([pid, target, condition, block, ([-1] if start == 0 else []) + pseudo_ids])
    return pd.DataFrame(rows, columns=["pid", "target", "condition", "block", "pseudo_ids"])


def spike_time_remap(one, pid, eid, pname):
    """
    Function mapping the release spike times of a probe to the re-extracted ones.

    Parameters
    ----------
    one : one.api.One
        ONE instance.
    pid, eid, pname : str
        Probe insertion, session and probe name.

    Returns
    -------
    callable
        `t_shifted = f(t_original)`, session times (s).
    """
    rec = re.sub(r"[a-d]$", "", pname)
    sync_original = np.load(one.load_dataset(eid, "*.sync.npy", collection=f"raw_ephys_data/{rec}", download_only=True))
    sync_new = np.load(next(OUTPUT_PATH.joinpath(pid, "raw_ephys_data", rec).glob("*.sync.npy")))
    # same interpolation as ibllib.ephys.sync_probes.apply_sync
    to_probe = interp1d(sync_original[:, 1], sync_original[:, 0], fill_value="extrapolate")
    to_session = interp1d(sync_new[:, 0], sync_new[:, 1], fill_value="extrapolate")
    return lambda t: to_session(to_probe(t))


def run_task(one, task):
    """
    Run the BWM decoding of one array task.

    Parameters
    ----------
    one : one.api.One
        ONE instance.
    task : pandas.Series
        Row of `tasks()`.

    Returns
    -------
    list of pathlib.Path
        Pickles written by `fit_session_ephys`, one per region.
    """
    patch_aggregate_tables()
    eid, pname = PIDS.set_index("pid").loc[task.pid, ["eid", "pname"]]  # no pid2eid in local mode on popeye
    bwm_df = bwm_query(one=one, freeze="2023_12_bwm_release")
    subject = bwm_df[bwm_df.eid == eid].subject.unique()[0]
    probe_name = list(bwm_df[bwm_df.eid == eid].probe_name)
    probe_name = probe_name[0] if len(probe_name) == 1 else probe_name
    load_good_units = brainwidemap.bwm_loading.load_good_units
    if task.condition == "shifted":
        remap = spike_time_remap(one, task.pid, eid, pname)

        def load_good_units(*args, **kwargs):
            """`load_good_units` with the spike times of the affected probe remapped."""
            spikes, clusters = brainwidemap.bwm_loading.load_good_units(*args, **kwargs)
            if kwargs.get("pname") == pname:
                spikes["times"] = remap(spikes["times"])
            return spikes, clusters

    # prepare_data imported the name: patch it there, the original is restored for the next task of the process
    prior_localization.prepare_data.load_good_units = load_good_units
    align_event, time_window, saturation_intervals, model, estimator, min_trials = BWM_TARGETS[task.target]
    config = check_config({"estimator": estimator})
    config["min_units"] = 1
    config["min_trials"] = min_trials
    config["estimator_kwargs"]["max_iter"] = 20000
    config["balanced_weighting"] = estimator == "LogisticRegression"
    # prepare_behavior reads the module config of prepare_data (estimator of config.yml, LogisticRegression), not the
    # one passed: its two-class check on the pseudo-session targets would reject a continuous target such as pLeft
    prior_localization.prepare_data.config["estimator"] = config["estimator"]
    return fit_session_ephys(
        one, eid, subject, probe_name, output_dir=DECODING_PATH.joinpath(task.condition, task.target, f"block_{task.block}"),
        pseudo_ids=task.pseudo_ids, target=task.target, align_event=align_event, min_rt=0.08, max_rt=2.0,
        time_window=time_window, saturation_intervals=saturation_intervals, model=model, n_runs=10, binsize=None,
        n_bins_lag=None, n_bins=None, compute_neurometrics=False, motor_residuals=False, config=config,
    )  # fmt: skip


def read_pickle_scores(file):
    """
    Scores of one prior-localization pickle: mean over runs per pseudo id, or None if being written.

    Parameters
    ----------
    file : pathlib.Path
        `decoding/<condition>/<target>/block_<k>/<subject>/<eid>/<region>_<probe>_pseudo_ids_<first>_<last>.pkl`.

    Returns
    -------
    dict or None
    """
    try:
        with open(file, "rb") as fid:
            res = pickle.load(fid)
    except (EOFError, pickle.UnpicklingError):  # being written by a running task
        return None
    scores = pd.DataFrame([{"pseudo_id": f["pseudo_id"], "score": f["scores_test_full"]} for f in res["fit"]])
    return {
        "condition": file.parts[-6], "target": file.parts[-5], "eid": res["eid"], "region": "+".join(res["region"]),
        "probe": "+".join(np.atleast_1d(res["probe"])), "n_units": int(res["N_units"]), "file": file.name,
        "scores": scores.groupby("pseudo_id").score.mean(),
    }  # fmt: skip


def aggregate(workers=1):
    """
    Score, null median and p-value per session x region x target x condition (prior-localization `results.py` stage 2).

    The score of a pseudo id is the mean over the 10 runs (balanced accuracy, R2 for pLeft). The p-value counts the real
    score among the nulls: `mean(concat(null, [real]) >= real)`. Writes `decode.pqt` in `OUTPUT_PATH`. The pickles hold
    the trial tables of every fit and are slow to read from ceph: `workers` processes read them in parallel.
    """
    eid2pid = dict(zip(PIDS.eid, PIDS.pid))
    files = sorted(DECODING_PATH.glob("*/*/block_*/*/*/*.pkl"))
    rows = joblib.Parallel(n_jobs=workers)(joblib.delayed(read_pickle_scores)(f) for f in files)
    print(f"{len(files)} pickles, {sum(r is None for r in rows)} skipped (being written)", flush=True)
    df = pd.DataFrame([r for r in rows if r is not None])
    # pseudo-session blocks of the same session x region are in separate pickles
    out = []
    for key, g in df.groupby(["condition", "target", "eid", "region"]):
        s = pd.concat(g.scores.tolist())
        real, null = s.get(-1, np.nan), s[s.index > 0].values
        out.append(
            dict(zip(["condition", "target", "eid", "region"], key))
            | {
                "pid": eid2pid.get(key[2]), "n_units": g.n_units.iloc[0], "n_pseudo": null.size, "score": real,
                "median_null": np.median(null), "p_value": np.mean(np.r_[null, real] >= real),
            }
        )  # fmt: skip
    out = pd.DataFrame(out)
    out["score_minus_null"] = out.score - out.median_null
    out.to_parquet(OUTPUT_PATH.joinpath("decode.pqt"))
    print(out.pivot_table(index=["target", "region"], columns="condition", values="score_minus_null").round(3).to_string())


_one = None


def run_task_safe(i):
    """Run task `i` in a worker process if not done, with its own ONE instance; log failures instead of raising."""
    global _one
    file_done, file_error = DECODING_PATH.joinpath("_done", str(i)), DECODING_PATH.joinpath("_errors", f"{i}.error")
    if file_done.exists():
        return
    file_done.parent.mkdir(parents=True, exist_ok=True)
    file_error.parent.mkdir(parents=True, exist_ok=True)
    _one = _one or ONE(mode="local")
    task = tasks().iloc[i]
    try:
        run_task(_one, task)
        file_done.touch()
        file_error.unlink(missing_ok=True)
        print(i, task.pid, task.target, task.condition, task.block, "done", flush=True)
    except Exception:  # noqa: BLE001
        file_error.write_text(traceback.format_exc())
        print(i, task.pid, task.target, task.condition, task.block, "ERROR", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", action="store_true", help="download the BWM trials table")
    group.add_argument("--task", type=int, help="single task index, 0-based")
    group.add_argument("--node", type=int, help="array element: runs tasks[node::n_nodes] in parallel")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=46)
    args = parser.parse_args()
    if args.stage or args.node is not None:
        # download the Allen atlas (SpikeSortingLoader) in one process: concurrent first downloads by the workers
        # read each other's half-written files (BadZipFile, EOFError)
        from iblatlas.atlas import AllenAtlas

        AllenAtlas(25)
    if args.stage:
        BWM_TABLES.mkdir(parents=True, exist_ok=True)
        print(brainwidemap.bwm_loading.download_aggregate_tables(ONE(), type="trials"))
        print(f"{len(tasks())} array tasks")
    elif args.node is not None:
        todo = list(range(args.node, len(tasks()), args.n_nodes))
        print(f"node {args.node}/{args.n_nodes}: {len(todo)} tasks on {args.workers} workers", flush=True)
        joblib.Parallel(n_jobs=args.workers)(joblib.delayed(run_task_safe)(i) for i in todo)
    elif args.aggregate:  # no ONE needed, can run at any time on the tasks done so far: sbatch on a node for speed
        aggregate(workers=args.workers)
    else:
        task = tasks().iloc[args.task]
        print(task.to_dict(), flush=True)
        print(run_task(ONE(mode="local"), task))
