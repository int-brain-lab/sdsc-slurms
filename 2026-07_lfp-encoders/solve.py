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


def _mask_targets(acc: Accumulator, mask: np.ndarray) -> Accumulator:
    """Restrict an accumulator to a target (``Y`` column) subset.

    ``xsum``/``xtx``/``n`` are X-side statistics (shared by every target) and
    pass through unchanged; only the per-target fields (``ysum``/``ysq`` and
    ``xty``'s target axis) are sliced. Used by :func:`solve_encoding_grouped`
    to score each target-group's own fit from the one shared accumulator
    (no re-streaming).
    """
    return Accumulator(acc.n, acc.xsum, acc.ysum[mask], acc.ysq[mask], acc.xtx, acc.xty[:, mask])


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
    # Set only by solve_encoding_grouped: target-group (e.g. band) -> lambda. Unlike
    # `lam`'s dict form (regressor-group -> lambda, the X-side penalty in build_penalty),
    # this is keyed by *target* group, so results_io can map it onto scores by
    # target_meta["band"] unambiguously -- kept as a separate field rather than
    # overloading `lam` so the two dict meanings can never be confused.
    lam_by_group: dict[object, float] | None = None


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


def solve_encoding_grouped(
    design: Design,
    targets: Targets,
    lam_by_group: dict[object, float],
    target_groups: np.ndarray,
    n_folds: int = 5,
    chunk_samples: int = 250 * 300,
) -> EncodingResult:
    """Fit the encoding model with a separate lambda per *target* group (e.g. band).

    ``solve_encoding`` does one Cholesky solve shared by every target column,
    so one lambda has to serve all of them at once. This does one solve **per
    target group** instead, each with its own penalty -- e.g. gamma can be
    smoothed harder than delta without delta paying for it, or vice versa.
    Motivated by the median-lambda-selection collapse in ``select_lambda``
    (see ``PLAN.md``/``index.qmd`` "Result 5"): a single pooled lambda can be
    tuned well for the bulk of targets while badly under-regularising a
    minority group, and this is the fitting-side fix that actually *consumes*
    a per-group lambda (``select_lambda_robust(..., groups=...)`` only
    *picks* one; nothing before this function could use it).

    The shared streaming pass (``accumulate_folds``) runs once regardless of
    group count -- only the ``(n_cols, n_cols)`` Cholesky factorisation
    repeats per group, which is cheap.

    Parameters
    ----------
    design : Design
    targets : Targets
    lam_by_group : dict
        Group label -> lambda, e.g. from ``select_lambda_robust(design,
        targets, lambdas, groups=target_groups)``.
    target_groups : ndarray, shape (n_targets,)
        Per-target group label matching ``lam_by_group``'s keys and
        ``targets.target_meta``'s row order (e.g.
        ``targets.target_meta["band"].to_numpy()``). Every target must
        belong to a group present in ``lam_by_group``.
    n_folds, chunk_samples : see :func:`solve_encoding`.

    Returns
    -------
    EncodingResult
        Same shape as :func:`solve_encoding`'s output. ``lam`` is
        ``lam_by_group`` verbatim (a target-group dict) -- do not pass this
        into :func:`build_penalty`/:func:`solve_encoding` directly, that
        function's dict form means something different (per **regressor**
        group, not per target group).
    """
    target_groups = np.asarray(target_groups)
    if not set(np.unique(target_groups)) <= set(lam_by_group):
        raise ValueError("every target group must have a lambda in lam_by_group")

    accs = accumulate_folds(design, targets, n_folds, chunk_samples)
    total = _sum(accs)

    n_targets = targets.n_targets
    W = np.empty((design.n_cols, n_targets))
    a = np.empty(n_targets)
    r2_full = np.empty(n_targets)
    r2_cv = np.empty(n_targets)
    dr2 = {group: np.empty(n_targets) for group in design.groups}

    for g, lam in lam_by_group.items():
        mask = target_groups == g
        if not mask.any():
            continue
        P = build_penalty(design, lam)
        total_g = _mask_targets(total, mask)
        sxx_g, sxy_g, xbar_g, ybar_g = centred_cross(total_g)
        w_g, a_g = _fit_and_intercept(sxx_g, sxy_g, xbar_g, ybar_g, P)
        W[:, mask], a[mask] = w_g, a_g
        r2_full[mask] = _r2(total_g, w_g, a_g)

        accs_g = [_mask_targets(acc, mask) for acc in accs]
        r2_cv[mask] = _cv_r2(accs_g, design, lam)
        for group, idxs in design.groups.items():
            cols = np.r_[tuple(np.arange(r * design.n_basis, (r + 1) * design.n_basis) for r in idxs)]
            keep = np.setdiff1d(np.arange(design.n_cols), cols)
            dr2[group][mask] = r2_cv[mask] - _cv_r2(accs_g, design, lam, keep=keep)

    return EncodingResult(
        pid=design.pid, kind=targets.kind, lam=lam_by_group, n_folds=n_folds,
        W=W, intercept=a, kernels=_kernels(design, W), taus=design.taus,
        r2_full=r2_full, r2_cv=r2_cv, dr2=dr2, target_meta=targets.target_meta,
        base_names=design.base_names, groups=design.groups, lam_by_group=lam_by_group,
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


def permutation_null_r2_grouped(
    design: Design,
    targets: Targets,
    lam_by_group: dict[object, float],
    target_groups: np.ndarray,
    n_perm: int = 30,
    n_folds: int = 5,
    seed: int = 0,
    chunk_samples: int = 250 * 300,
) -> np.ndarray:
    """:func:`permutation_null_r2`, but scoring each target group with its own lambda.

    The null must use the **same** per-group lambda as the real fit
    (:func:`solve_encoding_grouped`) or the permutation p-value compares two
    differently-regularised fits instead of isolating alignment. Only the
    scoring step changes per group (via :func:`_mask_targets`); the shifted
    streaming pass is shared across every group, same cost as
    :func:`permutation_null_r2`.

    Parameters
    ----------
    design, targets : Design, Targets
    lam_by_group : dict
        Group label -> lambda, as returned by ``select_lambda_robust(...,
        groups=target_groups)`` and passed to :func:`solve_encoding_grouped`.
    target_groups : ndarray, shape (n_targets,)
        Per-target group label matching ``lam_by_group``'s keys.
    n_perm, n_folds, seed, chunk_samples : see :func:`permutation_null_r2`.

    Returns
    -------
    ndarray, shape (n_perm, n_targets)
    """
    rng = np.random.default_rng(seed)
    n = design.n_samples
    margin = int(round(10.0 * design.fs))
    target_groups = np.asarray(target_groups)
    null = np.zeros((n_perm, targets.n_targets))
    for i in range(n_perm):
        shift = int(rng.integers(margin, n - margin))
        shifted = replace(design, base=np.roll(design.base, shift, axis=0))
        accs = accumulate_folds(shifted, targets, n_folds, chunk_samples)
        for g, lam in lam_by_group.items():
            mask = target_groups == g
            if not mask.any():
                continue
            accs_g = [_mask_targets(acc, mask) for acc in accs]
            null[i, mask] = _cv_r2(accs_g, shifted, lam)
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


def select_lambda_robust(
    design: Design,
    targets: Targets,
    lambdas: np.ndarray,
    groups: np.ndarray | None = None,
    n_folds: int = 5,
    chunk_samples: int = 250 * 300,
    floor: float = -0.3,
    floor_quantile: float = 0.05,
) -> tuple[float | dict[object, float], np.ndarray | dict[object, np.ndarray]]:
    """Sweep ``lambda`` like :func:`select_lambda`, but pick a tail-safe winner.

    ``select_lambda``'s median objective is *blind* to a collapsing subset of
    targets: a lambda that lets a whole band overfit catastrophically can
    still have the best median if the remaining targets improve, because the
    median only looks at the middle of the distribution. That is exactly the
    failure this project's compression comparison exposed (see
    ``PLAN.md``/``index.qmd`` "Result 5") -- a single global lambda, chosen
    this way, occasionally under-regularises an entire insertion.

    Two changes, both applied to the *same* six-or-so candidate fits (no
    extra streaming/accumulation cost over ``select_lambda``):

    1. **Objective**: mean of per-target CV R² clipped to ``[-1, 1]`` instead
       of the raw median. A meaningful fraction of targets collapsing pulls
       a mean down sharply; the median can shrug off up to half the targets
       failing without moving.
    2. **Worst-case gate**: among candidate lambdas, only those whose
       ``floor_quantile`` (default 5th percentile) of CV R² clears
       ``floor`` are eligible to win. If none clear it, fall back to the
       *largest* lambda in the grid rather than the objective-best one --
       cheap insurance, since the R²(lambda) tuning curve is documented as
       nearly flat at this basis resolution (see index.qmd "Diagnostics"),
       so over-regularising here costs little while under-regularising can
       cost everything.

    Parameters
    ----------
    design : Design
    targets : Targets
    lambdas : ndarray
        Candidate penalty strengths (should span comfortably above the
        largest value ``select_lambda`` has ever picked, so the fallback has
        somewhere safe to land).
    groups : ndarray, shape (n_targets,), optional
        Per-target group label (e.g. ``targets.target_meta["band"].to_numpy()``).
        If given, a separate winning lambda is chosen **per group** from the
        same shared candidate fits -- so one group collapsing cannot be
        masked by another group's improvement, which a single pooled
        objective (grouped or not) cannot rule out on its own. If ``None``
        (default), one lambda is chosen for all targets together.
    n_folds, chunk_samples : see :func:`select_lambda`.
    floor : float, default -0.3
        Worst-case-quantile CV R² a candidate lambda must clear per group.
    floor_quantile : float, default 0.05
        Quantile of the group's per-target CV R² checked against ``floor``.

    Returns
    -------
    best : float or dict
        Selected penalty strength (or one per group if ``groups`` given).
    curve : ndarray or dict
        Objective value per candidate lambda (or one curve per group), for
        the diagnostic plot -- NOT the same scale as ``select_lambda``'s
        median curve (this one is a clipped mean).
    """
    accs = accumulate_folds(design, targets, n_folds, chunk_samples)
    all_r2 = np.stack([_cv_r2(accs, design, float(lam)) for lam in lambdas])  # (n_lambda, n_targets)
    return _pick_lambda_from_curve(all_r2, lambdas, groups, floor, floor_quantile)


def _pick_lambda_from_curve(
    all_r2: np.ndarray, lambdas: np.ndarray, groups: np.ndarray | None, floor: float, floor_quantile: float,
) -> tuple[float | dict[object, float], np.ndarray | dict[object, np.ndarray]]:
    """Selection logic behind :func:`select_lambda_robust`, factored out for unit testing.

    Parameters
    ----------
    all_r2 : ndarray, shape (n_lambda, n_targets)
        Held-out CV R² for every candidate lambda (already computed).
    lambdas, groups, floor, floor_quantile : see :func:`select_lambda_robust`.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    label = np.zeros(all_r2.shape[1], dtype=int) if groups is None else np.asarray(groups)
    best: dict[object, float] = {}
    curves: dict[object, np.ndarray] = {}
    for g in np.unique(label) if groups is not None else [0]:
        mask = label == g
        sub = all_r2[:, mask]  # (n_lambda, n_group_targets)
        objective = np.clip(sub, -1.0, 1.0).mean(axis=1)
        safe = np.quantile(sub, floor_quantile, axis=1) >= floor
        if safe.any():
            candidates = np.flatnonzero(safe)
            best_i = candidates[int(np.argmax(objective[candidates]))]
        else:
            best_i = len(lambdas) - 1  # no candidate clears the floor: fall back to the largest lambda
        best[g] = float(lambdas[best_i])
        curves[g] = objective

    if groups is None:
        return best[0], curves[0]
    return best, curves