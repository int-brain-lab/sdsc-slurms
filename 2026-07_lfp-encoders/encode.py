"""Brain-wide LFP-encoding fit — SDSC cluster driver.

Runs the lagged LFP←behaviour encoding model over all BWM insertions, once per
**LFP source**, to measure how lossy compression affects the recoverable
behaviour signal:

    --lfp-source default       lfpack SVD+WP (ε=150, α=28)   lf_compressed_all_bwm.h5
    --lfp-source aggressive    lfpack SVD+WP (ε=450, α=96)   lf_compressed_aggressive_all_bwm.h5
    --lfp-source uncompressed  Cadzow checkpoint (250 Hz CAR, pre-SVD/WP) cells/<pid>/lf_resampled_car_cadzow.npy

The two compressed tiers read from a single consolidated HDF5 archive each — the
BWM lfpack files published on S3 (``resources/ibl-agent-data/``), keyed by PID
(``recording=pid``). Download them once with ``--download`` before submitting.
The uncompressed reference is **always** the per-PID Cadzow checkpoint, never the
raw .cbin: it is the exact pre-compression signal on the same 250 Hz grid, so R²
differences across the three sources isolate the SVD+WP compression.

Behaviour is loaded **from ONE** (`OneSdsc` local mirror) via :mod:`behavior_one`
— not the bwm_behavior shards — so wheel is complete for every session (cf.
int-brain-lab/ibl-ai-agent#18). Shards are written under ``<outdir>/<source>/`` so
the three runs never collide. Fit/scoring/persistence reuse the analysis modules
copied in beside this script (``design`` / ``targets`` / ``solve`` / ``results_io`` /
``lfpack_io``).

One 48-core node per ``--array`` task; each task fits its stripe of PIDs
(``task_id::array_count``) with a joblib pool. Resumable: a PID whose band shard
already exists is skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import joblib
import numpy as np

# the shared science modules live in this dir; make imports work from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))
import behavior_one
import design as design_mod
import lfpack_io as io
import results_io as rio
import solve as solve_mod
import targets as targets_mod

# ── SDSC paths (confirm on the cluster) ─────────────────────────────────────────
# Local root for the consolidated compressed lfpack archives fetched from S3
# (flat, one .h5 per tier — see --download).
LFP_DATA_ROOT = Path("/mnt/home/owinter/ceph/lfp-encoders")
# Per-PID Cadzow checkpoints (the uncompressed reference) from the 2026-06-lfpack job.
LFP_CELLS_ROOT = Path("/mnt/home/owinter/ceph/ea/denoised_lfp")
OUTPUT_ROOT = Path("/mnt/home/owinter/ceph/lfp-encoders/results_bwm")

UNCOMPRESSED = "uncompressed"
CADZOW_NPY = "lf_resampled_car_cadzow.npy"  # per-PID uncompressed reference (never raw .cbin)

# Compressed tiers -> consolidated BWM archive published on the IBL public S3 bucket
# (ibl-brain-wide-map-public), keyed by PID (recording=pid).
S3_LFP_ROOT = "resources/ibl-agent-data"
COMPRESSED_FILES = {
    "default": "lf_compressed_all_bwm.h5",
    "aggressive": "lf_compressed_aggressive_all_bwm.h5",
}
SOURCES = (*COMPRESSED_FILES, UNCOMPRESSED)

LAMBDAS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])  # 1e5 headroom: some PIDs already sat at the old 1e4 ceiling
N_BASIS = 10


# One OneSdsc per worker process, reused across every PID that worker handles.
# OneSdsc subclasses OneAlyx: constructing it authenticates and loads cache tables,
# so a fresh instance per PID would stampede alyx. Keyed on the worker's os.getpid().
_ONE_CACHE: dict[int, object] = {}


def _make_one(stagger: float = 0.0):
    """Return this worker's SDSC ONE, building it once and reusing it thereafter.

    Reads the on-cluster IBL mirror (no downloads). The first construction in each
    worker process sleeps a random delay in ``[0, stagger)`` seconds so the workers'
    initial alyx handshakes spread out instead of all landing at job start; cached
    reuse afterwards means each worker connects only once.
    """
    key = os.getpid()
    one = _ONE_CACHE.get(key)
    if one is None:
        if stagger > 0:
            time.sleep(random.SystemRandom().uniform(0.0, stagger))
        from deploy.iblsdsc import OneSdsc as ONE

        one = ONE()
        _ONE_CACHE[key] = one
    return one


def compressed_h5(source: str) -> Path:
    """Local path to the consolidated compressed lfpack archive for ``source``."""
    return LFP_DATA_ROOT.joinpath(COMPRESSED_FILES[source])


def download_lfp(source: str) -> Path:
    """Fetch the consolidated compressed archive for ``source`` from S3 if missing.

    Uses the ONE public-S3 helper (IBL public bucket, no AWS credentials); it checks
    the file size and skips the transfer when the local copy is already complete. Not
    safe to run concurrently on the same file, so it is a pre-submit step
    (``--download``), never part of the array fit.
    """
    from one.remote.aws import s3_download_file

    dst = compressed_h5(source)
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3_download_file(source=f"{S3_LFP_ROOT}/{COMPRESSED_FILES[source]}", destination=str(dst))
    return dst


def _grid_source(source: str) -> str:
    """Source whose lfpack archive supplies the aligned time base + channel meta.

    The three tiers share one 250 Hz grid and channel layout, so the uncompressed
    reference borrows both from the default archive.
    """
    return "default" if source == UNCOMPRESSED else source


def available_pids(source: str) -> list[str]:
    """PIDs fittable for ``source`` (the recording universe).

    Compressed tiers list the PIDs packed into the consolidated archive. The
    uncompressed tier intersects the on-disk Cadzow checkpoints with the default
    archive's PIDs: it borrows that archive's grid/channels (so a PID absent from it
    can't be fitted anyway), and the intersection keeps all three tiers over the same
    BWM PID set for a like-for-like comparison. ``cells/`` holds non-BWM checkpoints
    too, so the raw glob (~1099) is a superset of the ~700 BWM insertions.
    """
    from lfpack import LFPackReader

    if source == UNCOMPRESSED:
        cadzow = {p.parent.name for p in LFP_CELLS_ROOT.glob(f"*/{CADZOW_NPY}")}
        archive = set(LFPackReader.recordings(str(compressed_h5("default"))))
        return sorted(cadzow & archive)
    return sorted(LFPackReader.recordings(str(compressed_h5(source))))


def read_uncompressed(pid: str, bin_channels: int = io.BIN_CHANNELS):
    """Read the uncompressed 250 Hz CAR reference (Cadzow checkpoint) for one PID.

    Voltages come from ``lf_resampled_car_cadzow.npy`` (pre-SVD/WP, never the raw
    .cbin); the aligned time base and channel metadata are borrowed from the default
    lfpack archive so the grid matches the compressed tiers exactly. Returns the same
    tuple as ``targets.read_full_lfp``.
    """
    io.LFP_H5 = compressed_h5("default")
    reader = io.open_lfp(pid, bin_channels=bin_channels)
    try:
        fs, nc = float(reader.fs), reader.nc
        channels = io.channels_frame(reader)
        tvec = np.asarray(reader.times, dtype=np.float64)
    finally:
        reader.close()

    arr = np.asarray(np.load(LFP_CELLS_ROOT.joinpath(pid, CADZOW_NPY), mmap_mode="r"))
    if arr.shape[0] < arr.shape[1]:  # (channels, samples) -> (samples, channels)
        arr = arr.T
    # bin adjacent electrodes to the archive's target count (matches read_full_lfp)
    group = arr.shape[1] // nc
    volts = arr[:, : nc * group].reshape(arr.shape[0], nc, group).sum(axis=2).astype(np.float32)
    m = min(volts.shape[0], tvec.size)  # align sample count to the archive grid
    return volts[:m], channels, fs, tvec[:m]


def make_targets_for(pid: str, source: str, kind: str) -> targets_mod.Targets:
    """Build LFP targets for one PID from the requested source.

    Compressed sources reuse ``targets.make_targets`` by pointing ``lfpack_io`` at the
    per-PID archive; the uncompressed source feeds the Cadzow checkpoint arrays through
    the same transforms via ``targets.targets_from_lfp``.
    """
    if source == UNCOMPRESSED:
        volts, channels, fs, tvec = read_uncompressed(pid)
        return targets_mod.targets_from_lfp(pid, kind, volts, channels, fs, tvec)
    io.LFP_H5 = compressed_h5(source)  # process-local under loky; open_lfp reads this
    return targets_mod.make_targets(pid, kind=kind)


def build_design(pid: str, eid: str, source: str, one) -> design_mod.Design:
    """Assemble the lagged design: LFP grid from ``source`` + behaviour from ONE.

    Mirrors ``design.make_design`` but sources trials/wheel/pupil from ONE and the
    time base from the (grid) LFP archive, gating wheel/pupil per availability.
    """
    io.LFP_H5 = compressed_h5(_grid_source(source))
    reader = io.open_lfp(pid)
    try:
        tvec = np.asarray(reader.times, dtype=np.float64)
        fs = float(reader.fs)
    finally:
        reader.close()

    trials = behavior_one.load_trials_one(eid, one)
    cont = behavior_one.load_continuous_one(eid, one)
    base, names, groups = design_mod.build_base_regressors(trials, cont, tvec)
    B, taus, n_pre, n_post = design_mod.raised_cosine_basis(fs, 1.5, 1.5, N_BASIS)

    n_bas = B.shape[1]
    col_slices = {g: slice(idx[0] * n_bas, (idx[-1] + 1) * n_bas) for g, idx in groups.items()}
    col_names = [f"{nm}@b{k}" for nm in names for k in range(n_bas)]
    return design_mod.Design(
        pid=pid, eid=eid, fs=fs, tvec=tvec, base=base, base_names=names,
        groups=groups, B=B, taus=taus, n_pre=n_pre, n_post=n_post,
        col_slices=col_slices, col_names=col_names,
    )


# Saturated-sample row exclusion --------------------------------------------------
# No padding by default: a real-trace check (RMS around saturation edges, plus an
# end-to-end fit/lambda comparison across 0s/7-sample/6.656s margins on PID
# 1a276285-8b0e-4cc9-9f0a-a3a002978724) found no measurable filter-transient leakage
# past the stored interval, while padding costs fold-balance for no benefit.
SATURATION_MARGIN_S = 0.0

# Well-posedness floor for a single CV fold after row exclusion: below this, a fold's
# Sxx is too thin relative to the design's column count for a trustworthy ridge solve.
# design.n_cols is ~90 (9 base regressors x 10 basis functions) for this project, so
# this floor (~2000 samples = 8 s at 250 Hz) is generous, not tight -- it exists only
# to catch the pathological case of a fold's window landing almost entirely inside a
# padded saturated span, not to police ordinary fold-to-fold imbalance.
MIN_VALID_PER_FOLD = 2000


def saturation_valid_mask(
    pid: str, tvec: np.ndarray, source: str, margin_s: float = SATURATION_MARGIN_S
) -> np.ndarray:
    """Boolean row mask over ``tvec`` excluding saturated LFP spans (+ padding).

    Reads the saturation table from ``source``'s own grid archive (via
    :func:`_grid_source`, same as :func:`build_design`) -- verified scale-independent
    (identical across default/aggressive for a given PID), so this is just reusing
    whichever archive this run already requires, not a new dependency (e.g. a
    standalone ``aggressive`` run never needs the ``default`` archive downloaded).
    Each stored interval is expanded by ``margin_s`` on both sides before exclusion,
    since the stored boundaries are an approximate detection (see
    :data:`SATURATION_MARGIN_S`).

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    tvec : ndarray, shape (n_samples,)
        Session-clock sample times (``Design.tvec``).
    source : str
        LFP source being fit (``"default"``, ``"aggressive"`` or ``"uncompressed"``).
    margin_s : float, default :data:`SATURATION_MARGIN_S`
        Padding applied to each side of every stored saturation interval. Pass ``0.0``
        to exclude only the stored interval itself (no padding).

    Returns
    -------
    ndarray of bool, shape (n_samples,)
        ``True`` = keep, ``False`` = excluded (saturated span + margin).
    """
    io.LFP_H5 = compressed_h5(_grid_source(source))
    reader = io.open_lfp(pid)
    try:
        sat = reader.saturation_times()
    finally:
        reader.close()

    valid = np.ones(tvec.shape[0], dtype=bool)
    if sat.empty:
        return valid
    starts = sat["start_time"].to_numpy() - margin_s
    stops = sat["stop_time"].to_numpy() + margin_s
    lo = np.searchsorted(tvec, starts, side="left")
    hi = np.searchsorted(tvec, stops, side="right")
    for a, b in zip(lo, hi):
        valid[a:b] = False
    return valid


def check_fold_coverage(valid: np.ndarray, n_folds: int, min_valid: int = MIN_VALID_PER_FOLD) -> None:
    """Raise if any contiguous CV fold would retain too few valid rows.

    Uses the same fold-edge convention as :func:`solve.accumulate_folds`
    (``np.linspace`` over the sample range) so this reflects the actual folds the fit
    will use. A big saturated span landing mostly inside one fold's window is exactly
    the scenario that can leave a fold too thin for a well-posed ridge solve -- this is
    the QC gate for that failure mode (analogous to ``design.MIN_CONTINUOUS_COVERAGE``
    gating wheel/pupil), distinct from ordinary fold-to-fold imbalance, which
    ``_cv_r2``/``_r2`` already handle generically via each accumulator's own ``n``.

    Raises
    ------
    ValueError
        If any fold's valid-row count is below ``min_valid``.
    """
    edges = np.linspace(0, valid.shape[0], n_folds + 1).astype(int)
    counts = [int(valid[edges[f]:edges[f + 1]].sum()) for f in range(n_folds)]
    if min(counts) < min_valid:
        raise ValueError(
            f"fold too thin after saturation exclusion: per-fold valid counts {counts}, "
            f"min {min_valid} required"
        )


def fit_pid(pid: str, outdir: Path, source: str, n_perm: int, n_folds: int,
            overwrite: bool, stagger: float = 0.0, lambda_mode: str = "per-band",
            saturation_margin_s: float | None = SATURATION_MARGIN_S) -> dict:
    """Fit and persist both target families (band, raw) for one PID and source.

    Returns a small status dict; large arrays are written to disk and freed here.
    ``stagger`` only bites on the worker's first PID (see :func:`_make_one`).

    Parameters
    ----------
    lambda_mode : {"per-band", "pooled"}, default "per-band"
        ``"per-band"`` selects and fits a separate lambda per target group
        (``band`` for ``kind="band"``; degrades to a single, no-op group for
        ``kind="raw"``) via ``select_lambda_robust``/``solve_encoding_grouped``
        -- fixes the collapse documented in PLAN.md, where one pooled,
        median-selected lambda occasionally under-regularises an entire
        insertion (the selection criterion is blind to a collapsing target
        subset, and a single lambda then has to serve every band at once).
        ``"pooled"`` is the original behaviour (``select_lambda``/
        ``solve_encoding``), kept for direct A/B comparison against the
        already-archived ``results_bwm_cluster`` runs.
    saturation_margin_s : float or None, default :data:`SATURATION_MARGIN_S`
        Padding (seconds) applied on each side of every stored saturation interval
        before excluding those rows from the fit (see :func:`saturation_valid_mask`).
        ``None`` disables row exclusion entirely (pre-exclusion behaviour, every
        sample kept) -- exposed for A/B comparison against different margins.
    """
    # resume only when *both* families are on disk: band is saved before raw, so a
    # PID interrupted between the two would otherwise be skipped with raw missing.
    scores = outdir.joinpath("scores")
    if not overwrite and all(scores.joinpath(f"{pid}_{k}.parquet").exists() for k in ("band", "raw")):
        return {"pid": pid, "status": "skip"}
    try:
        one = _make_one(stagger)
        eid, _ = one.pid2eid(pid)
        dsg = build_design(pid, eid, source, one)
        if saturation_margin_s is None:
            valid = None
        else:
            valid = saturation_valid_mask(pid, dsg.tvec, source, margin_s=saturation_margin_s)
            check_fold_coverage(valid, n_folds)
        for kind in ("band", "raw"):
            tgt = make_targets_for(pid, source, kind)
            if lambda_mode == "per-band":
                groups = tgt.target_meta["band"].to_numpy()
                lam, _ = solve_mod.select_lambda_robust(dsg, tgt, LAMBDAS, groups=groups, n_folds=n_folds, valid=valid)
                res = solve_mod.solve_encoding_grouped(dsg, tgt, lam, groups, n_folds=n_folds, valid=valid)
                null = solve_mod.permutation_null_r2_grouped(dsg, tgt, lam, groups, n_perm=n_perm, n_folds=n_folds, valid=valid)
            else:
                lam, _ = solve_mod.select_lambda(dsg, tgt, LAMBDAS, n_folds=n_folds, valid=valid)
                res = solve_mod.solve_encoding(dsg, tgt, lam=lam, n_folds=n_folds, valid=valid)
                null = solve_mod.permutation_null_r2(dsg, tgt, n_perm=n_perm, lam=lam, n_folds=n_folds, valid=valid)
            rio.save_pid_result(res, outdir, null=null)
            del tgt, res, null
        return {"pid": pid, "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - record and continue the batch
        return {"pid": pid, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain-wide LFP-encoding fit (SDSC)")
    parser.add_argument("--lfp-source", choices=list(SOURCES), default="default",
                        help="which LFP source to fit (run all three to compare)")
    parser.add_argument("--download", action="store_true",
                        help="fetch the consolidated compressed archive(s) from S3 and exit")
    parser.add_argument("--outdir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8, help="joblib PIDs in flight (band Y ~1.4 GB each)")
    parser.add_argument("--n-perm", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="fit only the first N PIDs (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stagger", type=float, default=30.0,
                        help="max seconds of random delay before each worker's first ONE connection")
    parser.add_argument("--lambda-mode", choices=["per-band", "pooled"], default="per-band",
                        help="per-band (default): separate lambda per band, fixes the pooled-median "
                             "collapse (see PLAN.md); pooled: original one-lambda-for-everything "
                             "behaviour, for A/B comparison against results_bwm_cluster")
    parser.add_argument("--saturation-margin-s", type=float, default=SATURATION_MARGIN_S,
                        help="padding (s) around each stored saturation interval before excluding "
                             "those rows from the fit (default: 0.0, no padding -- see "
                             "SATURATION_MARGIN_S). Pass a negative value to disable row exclusion "
                             "entirely (pre-exclusion behaviour, every sample kept).")
    args = parser.parse_args()
    saturation_margin_s = None if args.saturation_margin_s < 0 else args.saturation_margin_s

    # one-time pre-submit download of the consolidated archive(s) this source reads.
    # The uncompressed tier still needs the default archive for its grid + channel meta.
    if args.download:
        for src in (["default"] if args.lfp_source == UNCOMPRESSED else [args.lfp_source]):
            print(f"downloading {COMPRESSED_FILES[src]} …", flush=True)
            print(f"  -> {download_lfp(src)}", flush=True)
        return

    # fail fast (before fanning out) if the archive is not on disk yet
    grid = _grid_source(args.lfp_source)
    if not compressed_h5(grid).exists():
        raise SystemExit(
            f"missing {compressed_h5(grid)}; run "
            f"`python encode.py --download --lfp-source {args.lfp_source}` first"
        )

    # per-source output dir so the three runs never collide
    outdir = args.outdir.joinpath(args.lfp_source)
    outdir.mkdir(parents=True, exist_ok=True)

    pids = available_pids(args.lfp_source)
    if args.limit:
        pids = pids[: args.limit]
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    mine = pids[task_id::task_count]
    print(f"[{args.lfp_source}] task {task_id}/{task_count}: {len(mine)}/{len(pids)} PIDs, "
          f"{args.workers} workers", flush=True)

    # shared basis/config once (task 0 only, from the first fittable PID)
    if task_id == 0 and not outdir.joinpath("basis.npz").exists():
        one = _make_one()
        eid, _ = one.pid2eid(mine[0])
        rio.save_shared(build_design(mine[0], eid, args.lfp_source, one), outdir, targets_mod.BANDS,
                         extra={"lambda_mode": args.lambda_mode, "lambdas": LAMBDAS.tolist(),
                                "saturation_margin_s": saturation_margin_s})

    results = joblib.Parallel(n_jobs=args.workers, backend="loky")(
        joblib.delayed(fit_pid)(pid, outdir, args.lfp_source, args.n_perm, args.n_folds,
                                args.overwrite, args.stagger, args.lambda_mode, saturation_margin_s)
        for pid in mine
    )
    ok = sum(r["status"] == "ok" for r in results)
    err = [r for r in results if r["status"] == "error"]
    print(f"[{args.lfp_source}] task {task_id}: {ok} ok, "
          f"{sum(r['status']=='skip' for r in results)} skipped, {len(err)} errored", flush=True)
    for r in err:
        print(f"  ERROR {r['pid']}: {r['error']}", flush=True)


if __name__ == "__main__":
    main()