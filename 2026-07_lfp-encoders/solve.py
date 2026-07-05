"""Out-of-core solver for the lagged LFP-encoding model.

The full design ``X`` is never held whole; instead time-chunks (expanded on
demand by :meth:`design.Design.expand_range`) are streamed and reduced to
sufficient statistics -- ``XᵀX``, ``XᵀY`` and the sums / ``Σy²`` needed to
centre the fit and score it -- all accumulated in **float64** (the normal
equations square the condition number, so f32 would lose precision).

From those statistics the model is a smoothness-regularised ridge:
``(Sxx + P) W = Sxy``, where ``Sxx``/``Sxy`` are the centred cross-products and
``P`` is a block-diagonal second-difference (Tikhonov) penalty -- one block per
base regressor, scaled by a per-group ``lambda`` -- solved with a Cholesky
factorisation.

K-fold cross-validation is free: one accumulator per contiguous time-fold, with
the training set for a held-out fold being the sum of the others. Every score
(held-out R², per-group drop-R²) is computed from the fold accumulators via the
same closed form, so no predictions are ever materialised.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from design import Design
from targets import Targets


# --- sufficient statistics -------------------------------------------------
@dataclass
class Accumulator:
    """Float64 sufficient statistics over a set of samples.

    ``xtx``/``xty`` are the raw (uncentred) cross-products; ``xsum``/``ysum`` and
    ``ysq`` (``Σy²`` per target) let us centre and score without a second pass.
    """

    n: int
    xsum: np.ndarray  # (p,)
    ysum: np.ndarray  # (m,)
    ysq: np.ndarray  # (m,)
    xtx: np.ndarray  # (p, p)
    xty: np.ndarray  # (p, m)

    @classmethod
    def zeros(cls, p: int, m: int) -> "Accumulator":
        return cls(0, np.zeros(p), np.zeros(m), np.zeros(m), np.zeros((p, p)), np.zeros((p, m)))

    def add_chunk(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fold one aligned ``(X, Y)`` time-chunk into the statistics."""
        Xd, Yd = X.astype(np.float64), Y.astype(np.float64)
        self.n += Xd.shape[0]
        self.xsum += Xd.sum(0)
        self.ysum += Yd.sum(0)
        self.ysq += np.einsum("tm,tm->m", Yd, Yd)
        self.xtx += Xd.T @ Xd
        self.xty += Xd.T @ Yd

    def __add__(self, other: "Accumulator") -> "Accumulator":
        return Accumulator(
            self.n + other.n, self.xsum + other.xsum, self.ysum + other.ysum,
            self.ysq + other.ysq, self.xtx + other.xtx, self.xty + other.xty,
        )


def _sum(accs: list[Accumulator]) -> Accumulator:
    total = accs[0]
    for a in accs[1:]:
        total = total + a
    return total


def accumulate_folds(design: Design, targets: Targets, n_folds: int, chunk_samples: int) -> list[Accumulator]:
    """Stream the recording once into one accumulator per contiguous time-fold.

    Parameters
    ----------
    design : Design
        Provides the on-demand expanded design (``expand_range``).
    targets : Targets
        In-memory targets sliced per chunk.
    n_folds : int
        Number of contiguous cross-validation folds.
    chunk_samples : int
        Interior chunk length streamed at a time.

    Returns
    -------
    list of Accumulator
        One per fold, in temporal order.
    """
    if design.n_samples != targets.n_samples:
        raise ValueError("design and targets have different sample counts")
    p, m = design.n_cols, targets.n_targets
    edges = np.linspace(0, design.n_samples, n_folds + 1).astype(int)
    accs = []
    for f in range(n_folds):
        acc = Accumulator.zeros(p, m)
        for a in range(edges[f], edges[f + 1], chunk_samples):
            b = min(a + chunk_samples, edges[f + 1])
            acc.add_chunk(design.expand_range(a, b), targets.Y[a:b])
        accs.append(acc)
    return accs


# --- centring, penalty, fit ------------------------------------------------
def centred_cross(acc: Accumulator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centred ``(Sxx, Sxy, xbar, ybar)`` from raw statistics."""
    xbar, ybar = acc.xsum / acc.n, acc.ysum / acc.n
    sxx = acc.xtx - acc.n * np.outer(xbar, xbar)
    sxy = acc.xty - acc.n * np.outer(xbar, ybar)
    return sxx, sxy, xbar, ybar


def _second_diff_gram(m: int) -> np.ndarray:
    """``D2ᵀD2`` for the second-difference operator on ``m`` ordered coefficients."""
    if m < 3:
        return np.eye(m)  # too few coefficients to smooth -> plain ridge
    d2 = np.zeros((m - 2, m))
    for i in range(m - 2):
        d2[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d2.T @ d2


def build_penalty(design: Design, lam: float | dict[str, float]) -> np.ndarray:
    """Block-diagonal smoothness penalty ``P``, one block per base regressor.

    Parameters
    ----------
    design : Design
        Supplies column layout and group membership.
    lam : float or dict
        Global penalty strength, or per-group strengths keyed by group name
        (missing groups default to 0 = unpenalised).

    Returns
    -------
    ndarray, shape (n_cols, n_cols)
        The Tikhonov penalty added to ``Sxx`` before the Cholesky solve.
    """
    nb = design.n_basis
    base_group = {r: g for g, idxs in design.groups.items() for r in idxs}
    gram = _second_diff_gram(nb)
    P = np.zeros((design.n_cols, design.n_cols))
    for r, group in base_group.items():
        strength = lam.get(group, 0.0) if isinstance(lam, dict) else float(lam)
        s = slice(r * nb, (r + 1) * nb)
        P[s, s] = strength * gram
    return P


def ridge_fit(sxx: np.ndarray, sxy: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Solve ``(Sxx + P) W = Sxy`` for the weights ``W`` (p x m) via Cholesky."""
    c, low = cho_factor(sxx + P, lower=True)
    return cho_solve((c, low), sxy)


def _r2(acc: Accumulator, W: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-target R² of model ``(W, a)`` on ``acc``, from statistics alone.

    ``SS_res`` is expanded so no prediction vector is formed; the baseline is the
    accumulator's own per-target mean (held-out mean for CV folds).
    """
    n = acc.n
    quad = np.einsum("pm,pq,qm->m", W, acc.xtx, W)  # wᵀ XᵀX w
    w_xty = np.einsum("pm,pm->m", W, acc.xty)  # wᵀ XᵀY
    xsum_w = acc.xsum @ W  # Σx · w
    ss_res = acc.ysq - 2.0 * (a * acc.ysum + w_xty) + n * a**2 + 2.0 * a * xsum_w + quad
    ss_tot = acc.ysq - acc.ysum**2 / n
    return 1.0 - ss_res / ss_tot


def _fit_and_intercept(sxx, sxy, xbar, ybar, P):
    """Fit centred ridge and recover the implied intercept ``a = ȳ - x̄·W``."""
    W = ridge_fit(sxx, sxy, P)
    return W, ybar - xbar @ W


# --- orchestration ---------------------------------------------------------
@dataclass
class EncodingResult:
    """Fitted model, kernels and scores for one insertion / target family."""

    pid: str
    kind: str
    lam: float | dict[str, float]
    n_folds: int
    W: np.ndarray  # (n_cols, n_targets) full-data weights
    intercept: np.ndarray  # (n_targets,)
    kernels: dict[str, np.ndarray]  # base_name -> (n_lags, n_targets)
    taus: np.ndarray  # lag axis (s)
    r2_full: np.ndarray  # (n_targets,) in-sample
    r2_cv: np.ndarray  # (n_targets,) mean held-out
    dr2: dict[str, np.ndarray]  # group -> (n_targets,) CV drop-R²
    target_meta: object
    base_names: list[str] = field(default_factory=list)
    groups: dict[str, list[int]] = field(default_factory=dict)


def _cv_r2(accs: list[Accumulator], design: Design, lam, keep: np.ndarray | None = None) -> np.ndarray:
    """Mean held-out R² over folds; ``keep`` masks columns for reduced models."""
    p = design.n_cols
    P = build_penalty(design, lam)
    m = accs[0].ysum.size
    scores = np.zeros((len(accs), m))
    for f in range(len(accs)):
        train = _sum([a for i, a in enumerate(accs) if i != f])
        sxx, sxy, xbar, ybar = centred_cross(train)
        if keep is None:
            W, a = _fit_and_intercept(sxx, sxy, xbar, ybar, P)
        else:  # reduced model: fit on kept columns, scatter back with zeros
            w_k = ridge_fit(sxx[np.ix_(keep, keep)], sxy[keep], P[np.ix_(keep, keep)])
            W = np.zeros((p, m))
            W[keep] = w_k
            a = ybar - xbar @ W
        scores[f] = _r2(accs[f], W, a)
    return scores.mean(0)


def _kernels(design: Design, W: np.ndarray) -> dict[str, np.ndarray]:
    """Map basis weights back to lag-domain kernels: ``K = B @ W_block`` per regressor."""
    nb = design.n_basis
    return {
        name: design.B @ W[r * nb:(r + 1) * nb]
        for r, name in enumerate(design.base_names)
    }


def solve_encoding(
    design: Design,
    targets: Targets,
    lam: float | dict[str, float] = 1.0,
    n_folds: int = 5,
    chunk_samples: int = 250 * 300,
) -> EncodingResult:
    """Fit the encoding model and score it out-of-core.

    Parameters
    ----------
    design : Design
        Lagged design for the insertion.
    targets : Targets
        Raw or band-power targets (same insertion, same ``tvec``).
    lam : float or dict, default 1.0
        Smoothness penalty (global or per-group).
    n_folds : int, default 5
        Contiguous cross-validation folds.
    chunk_samples : int, default 75000
        Streaming chunk length (~300 s at 250 Hz).

    Returns
    -------
    EncodingResult
        Full-data weights/kernels plus in-sample and cross-validated R² and
        per-group drop-R².
    """
    accs = accumulate_folds(design, targets, n_folds, chunk_samples)
    total = _sum(accs)
    P = build_penalty(design, lam)

    sxx, sxy, xbar, ybar = centred_cross(total)
    W, a = _fit_and_intercept(sxx, sxy, xbar, ybar, P)
    r2_full = _r2(total, W, a)
    r2_cv = _cv_r2(accs, design, lam)

    # per-group drop-R²: cross-validated loss from removing each group's columns
    dr2 = {}
    for group, idxs in design.groups.items():
        cols = np.r_[tuple(np.arange(r * design.n_basis, (r + 1) * design.n_basis) for r in idxs)]
        keep = np.setdiff1d(np.arange(design.n_cols), cols)
        dr2[group] = r2_cv - _cv_r2(accs, design, lam, keep=keep)

    return EncodingResult(
        pid=design.pid, kind=targets.kind, lam=lam, n_folds=n_folds,
        W=W, intercept=a, kernels=_kernels(design, W), taus=design.taus,
        r2_full=r2_full, r2_cv=r2_cv, dr2=dr2, target_meta=targets.target_meta,
        base_names=design.base_names, groups=design.groups,
    )


def permutation_null_r2(
    design: Design,
    targets: Targets,
    n_perm: int = 30,
    lam: float | dict[str, float] = 1.0,
    n_folds: int = 5,
    seed: int = 0,
    chunk_samples: int = 250 * 300,
) -> np.ndarray:
    """Circular-shift null distribution of cross-validated R² per target.

    The base regressors are rolled by a random offset (well beyond the kernel
    window) before each refit. This breaks the true stimulus/behaviour-to-LFP
    alignment while **preserving** the autocorrelation of both sides -- the
    oscillatory structure of the LFP and the temporal clustering of events -- so
    the resulting R² reflects what this model recovers by chance from
    time-series structure alone. A plain sample shuffle would destroy that
    autocorrelation and badly understate the true chance level.

    Parameters
    ----------
    design, targets : Design, Targets
        Model inputs (unchanged; a shifted copy of ``design`` is used per draw).
    n_perm : int, default 30
        Number of shifted refits.
    lam, n_folds, chunk_samples : see :func:`solve_encoding`.
    seed : int
        Seed for the shift offsets.

    Returns
    -------
    ndarray, shape (n_perm, n_targets)
        Held-out R² for each shifted refit; compare the observed R² against its
        column to get a per-target permutation p-value.
    """
    rng = np.random.default_rng(seed)
    n = design.n_samples
    margin = int(round(10.0 * design.fs))  # keep shift clear of the window edges
    null = np.zeros((n_perm, targets.n_targets))
    for i in range(n_perm):
        shift = int(rng.integers(margin, n - margin))
        shifted = replace(design, base=np.roll(design.base, shift, axis=0))
        accs = accumulate_folds(shifted, targets, n_folds, chunk_samples)
        null[i] = _cv_r2(accs, shifted, lam)
    return null


def select_lambda(
    design: Design,
    targets: Targets,
    lambdas: np.ndarray,
    n_folds: int = 5,
    chunk_samples: int = 250 * 300,
) -> tuple[float, np.ndarray]:
    """Sweep a global ``lambda`` and pick the one with best median held-out R².

    Returns
    -------
    best : float
        Selected penalty strength.
    curve : ndarray, shape (len(lambdas),)
        Median CV R² across targets for each lambda (for the diagnostic plot).
    """
    accs = accumulate_folds(design, targets, n_folds, chunk_samples)
    curve = np.array([np.median(_cv_r2(accs, design, float(lam))) for lam in lambdas])
    return float(lambdas[int(np.argmax(curve))]), curve