"""Persistence for encoding-model results.

Kernels are stored **compactly** as raised-cosine basis weights ``W`` (plus the
intercept), never as the expanded lag-domain arrays: since
``kernel = B @ W_block`` exactly, storing ``W`` with the shared basis is ~75x
smaller and lossless. The full kernels are reconstructed on demand with
:func:`expand_kernel`.

Everything is sharded per ``(pid, kind)`` so a brain-wide run's workers each
write only their own files (no write contention):

    <outdir>/model_config.json        shared model provenance
    <outdir>/basis.npz                shared B (n_lags x n_basis) + taus
    <outdir>/scores/<pid>_<kind>.parquet   tidy per-channel scores + null p-values
    <outdir>/kernels/<pid>_<kind>.npz      W (n_cols x n_targets), intercept, lam

:func:`load_scores` concatenates the score shards into one analysis-ready table
carrying depth, CCF coordinates and region for pooling across insertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from design import Design
from solve import EncodingResult


def permutation_pvalue(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Per-target permutation p-value with the standard ``+1`` correction.

    Parameters
    ----------
    observed : ndarray, shape (n_targets,)
        Observed held-out R² per target.
    null : ndarray, shape (n_perm, n_targets)
        Circular-shift null R² draws.

    Returns
    -------
    ndarray, shape (n_targets,)
        ``(1 + #{null >= observed}) / (n_perm + 1)``.
    """
    n_perm = null.shape[0]
    return (1 + (null >= observed[None, :]).sum(0)) / (n_perm + 1)


def save_shared(design: Design, outdir: Path, bands: dict, extra: dict | None = None) -> None:
    """Write the model config and shared basis (call once per run)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        "n_basis": design.n_basis,
        "n_pre": design.n_pre,
        "n_post": design.n_post,
        "fs": design.fs,
        "base_names": design.base_names,
        "groups": design.groups,
        "bands": {k: list(v) for k, v in bands.items()},
        "date": "2026-07-04",
        **(extra or {}),
    }
    outdir.joinpath("model_config.json").write_text(json.dumps(config, indent=2))
    np.savez(outdir.joinpath("basis.npz"), B=design.B, taus=design.taus)


def save_pid_result(
    result: EncodingResult, outdir: Path, null: np.ndarray | None = None
) -> tuple[Path, Path]:
    """Persist one fitted result as a score shard and a compact kernel file.

    Parameters
    ----------
    result : EncodingResult
        Fitted model for one ``(pid, kind)``.
    outdir : Path
        Run output directory (created if needed).
    null : ndarray, optional
        Circular-shift null R² draws; adds a ``p_value`` column when given.

    Returns
    -------
    (Path, Path)
        Paths to the written score parquet and kernel npz. The ``lam`` score column is
        the scalar penalty for a plain ``solve_encoding`` fit, that row's own band's
        penalty for a ``solve_encoding_grouped`` fit (via ``result.lam_by_group``), or
        NaN for the (currently unused in this pipeline) per-regressor-group penalty
        form of ``result.lam``.
    """
    outdir = Path(outdir)
    scores_dir = outdir.joinpath("scores")
    kernels_dir = outdir.joinpath("kernels")
    scores_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    scores = result.target_meta.copy()
    scores.insert(0, "pid", result.pid)
    scores.insert(1, "kind", result.kind)
    if result.lam_by_group is not None:
        scores["lam"] = scores["band"].map(result.lam_by_group).astype(float)
    else:
        scores["lam"] = float(result.lam) if not isinstance(result.lam, dict) else np.nan
    scores["r2_full"] = result.r2_full
    scores["r2_cv"] = result.r2_cv
    # Gated add-on groups vary per PID (~16 % lack wheel; pupil is coverage-gated);
    # flag availability so pooling can restrict to a consistent subset. Absent
    # groups leave their dr2_* column NaN after load_scores concatenation.
    scores["has_wheel"] = "wheel" in result.groups
    scores["has_pupil"] = "pupil" in result.groups
    for group, dr in result.dr2.items():
        scores[f"dr2_{group}"] = dr
    if null is not None:
        scores["p_value"] = permutation_pvalue(result.r2_cv, null)
        scores["null_p95"] = np.quantile(null, 0.95, axis=0)  # perm-count-independent threshold

    tag = f"{result.pid}_{result.kind}"
    score_path = scores_dir.joinpath(f"{tag}.parquet")
    kernel_path = kernels_dir.joinpath(f"{tag}.npz")
    scores.to_parquet(score_path, index=False)
    np.savez(
        kernel_path,
        W=result.W.astype(np.float32),
        intercept=result.intercept.astype(np.float32),
        lam=np.asarray(result.lam if not isinstance(result.lam, dict) else np.nan, dtype=float),
    )
    return score_path, kernel_path


def load_scores(outdir: Path) -> pd.DataFrame:
    """Concatenate all per-(pid, kind) score shards into one table."""
    shards = sorted(Path(outdir).joinpath("scores").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no score shards under {outdir}/scores")
    return pd.concat((pd.read_parquet(p) for p in shards), ignore_index=True)


def load_basis(outdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the shared basis ``(B, taus)``."""
    d = np.load(Path(outdir).joinpath("basis.npz"))
    return d["B"], d["taus"]


def load_kernels(pid: str, kind: str, outdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load compact weights ``(W, intercept)`` for one ``(pid, kind)``."""
    d = np.load(Path(outdir).joinpath("kernels", f"{pid}_{kind}.npz"))
    return d["W"], d["intercept"]


def expand_kernel(W: np.ndarray, base_index: int, B: np.ndarray) -> np.ndarray:
    """Reconstruct one regressor's lag-domain kernel from compact weights.

    Parameters
    ----------
    W : ndarray, shape (n_cols, n_targets)
        Compact basis weights (``n_cols == n_base * n_basis``).
    base_index : int
        Index of the base regressor (see ``base_names`` in the config).
    B : ndarray, shape (n_lags, n_basis)
        Shared basis matrix from :func:`load_basis`.

    Returns
    -------
    ndarray, shape (n_lags, n_targets)
        The kernel ``B @ W_block``.
    """
    nb = B.shape[1]
    return B @ W[base_index * nb:(base_index + 1) * nb]
