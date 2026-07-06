"""Lagged (FIR) design matrix for the LFP-encoding model.

The predictors are *base regressors* on the 250 Hz LFP grid -- event impulse
trains (optionally modulated by a per-trial covariate) and continuous behaviour
traces -- each convolved with a shared temporal basis so that a whole kernel is
fit per regressor. Two bases are offered:

- **raised cosine** (default): a handful of overlapping cosine bumps tiling the
  lag window, giving smooth kernels with few coefficients.
- **full FIR** (``n_basis=None``): one coefficient per lag sample, kept for
  validating the basis and for event-triggered-average comparisons.

The full lagged matrix ``X`` (up to ~100 GB for full FIR brain-wide) is never
materialised. The small base-regressor matrix (``n_samples x n_base``) is held
in memory; :meth:`Design.expand_range` convolves an arbitrary contiguous sample
range on demand, reaching into neighbouring samples so chunk boundaries carry no
edge loss. ``solve.py`` streams these chunks to accumulate ``X.T X`` / ``X.T Y``.

Lag convention: the kernel axis is the *delay* ``tau`` (seconds) of the
regressor relative to the LFP sample it explains. Positive ``tau`` = behaviour
precedes the LFP (causal history, ``t_post``); negative ``tau`` = behaviour
follows it (acausal, ``t_pre``), kept to expose alignment/sign errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

import lfpack_io as io


# --- temporal basis --------------------------------------------------------
def raised_cosine_basis(
    fs: float, t_pre: float = 1.5, t_post: float = 1.5, n_basis: int | None = 10
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build a lag basis over ``[-t_pre, +t_post]`` seconds.

    Parameters
    ----------
    fs : float
        Sampling rate of the regressor grid (Hz).
    t_pre, t_post : float
        Acausal and causal half-windows (seconds). ``t_post`` is the causal
        history over which behaviour may precede the LFP.
    n_basis : int or None, default 10
        Number of raised-cosine bumps. ``None`` selects the full-FIR identity
        basis (one column per lag sample).

    Returns
    -------
    B : ndarray, shape (n_lags, n_cols)
        Basis matrix; ``n_cols == n_basis`` (or ``n_lags`` for full FIR).
    taus : ndarray, shape (n_lags,)
        Delay of each lag sample in seconds (negative = acausal).
    n_pre, n_post : int
        Acausal / causal window lengths in samples.
    """
    n_pre = int(round(t_pre * fs))
    n_post = int(round(t_post * fs))
    taus = np.arange(-n_pre, n_post + 1)
    n_lags = taus.size

    if n_basis is None:  # full FIR: identity basis
        return np.eye(n_lags, dtype=np.float64), taus / fs, n_pre, n_post

    centres = np.linspace(0.0, n_lags - 1, n_basis)
    width = centres[1] - centres[0] if n_basis > 1 else float(n_lags)
    arg = (np.arange(n_lags)[:, None] - centres[None, :]) / width
    B = np.where(np.abs(arg) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * arg)), 0.0)
    return B, taus / fs, n_pre, n_post


# --- continuous-trace resampling ------------------------------------------
# Minimum fraction of the LFP session a continuous trace's own timestamps must span
# to be trusted as a regressor. Below this, np.interp's edge-clamping (see _resample)
# leaves most samples pinned to a single boundary value; _zscore then divides by the
# whole-trace std, which is deflated by that flat plateau and inflates the genuine
# variation in the covered window to non-physiological z-scores (observed z~25 for a
# session where the camera covered only the last 10%) -- a single such column can
# destabilise the shared ridge solve for every target at once (see PROMPT_sstot_zero.md).
MIN_CONTINUOUS_COVERAGE = 0.8


def _coverage(t_src: np.ndarray, tvec: np.ndarray) -> float:
    """Fraction of ``tvec``'s span actually covered by ``t_src``'s own time range."""
    span = tvec[-1] - tvec[0]
    if t_src.size == 0 or span <= 0:
        return 0.0
    covered = min(t_src[-1], tvec[-1]) - max(t_src[0], tvec[0])
    return max(0.0, covered) / span


def _resample(t_src: np.ndarray, v_src: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Linearly resample ``v_src(t_src)`` onto ``tvec``, ignoring non-finite input."""
    good = np.isfinite(t_src) & np.isfinite(v_src)
    return np.interp(tvec, t_src[good], v_src[good])


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardise so continuous regressors share a comparable scale for per-group lambda."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


# --- design container ------------------------------------------------------
@dataclass
class Design:
    """A streamable lagged design matrix for one insertion.

    Attributes
    ----------
    pid, eid : str
        Insertion and session ids.
    fs : float
        LFP/regressor sampling rate (Hz).
    tvec : ndarray, shape (n_samples,)
        Session-clock time of each grid sample (mirrors the LFP reader).
    base : ndarray, shape (n_samples, n_base), float32
        Base regressors before lag expansion.
    base_names : list of str
        Name of each base regressor.
    groups : dict[str, list[int]]
        Regressor group -> indices into ``base`` (used for grouped lambda / drop-R2).
    B : ndarray, shape (n_lags, n_basis)
        Temporal basis applied to every base regressor.
    taus : ndarray, shape (n_lags,)
        Lag axis in seconds.
    n_pre, n_post : int
        Acausal / causal window lengths in samples.
    col_slices : dict[str, slice]
        Group -> contiguous column span in the expanded matrix.
    col_names : list of str
        Name of each expanded column (``"<base>@b<k>"``).
    """

    pid: str
    eid: str
    fs: float
    tvec: np.ndarray
    base: np.ndarray
    base_names: list[str]
    groups: dict[str, list[int]]
    B: np.ndarray
    taus: np.ndarray
    n_pre: int
    n_post: int
    col_slices: dict[str, slice] = field(default_factory=dict)
    col_names: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.base.shape[0]

    @property
    def n_basis(self) -> int:
        return self.B.shape[1]

    @property
    def n_cols(self) -> int:
        return self.base.shape[1] * self.n_basis

    def expand_range(self, a: int, b: int) -> np.ndarray:
        """Expand the lagged design for the sample range ``[a, b)``.

        Reaches ``n_post`` samples before ``a`` and ``n_pre`` after ``b`` into the
        in-memory base regressors (zero-padded at the recording edges), so the
        returned rows are exact regardless of chunk boundaries.

        Parameters
        ----------
        a, b : int
            Half-open interior sample range.

        Returns
        -------
        ndarray, shape (b - a, n_cols), float32
            Expanded design; column order matches ``col_names`` / ``col_slices``.
        """
        n = self.n_samples
        lo, hi = a - self.n_post, b + self.n_pre  # base samples the window touches
        pad_lo, pad_hi = max(0, -lo), max(0, hi - n)
        win = self.base[max(0, lo):min(n, hi)]
        if pad_lo or pad_hi:
            win = np.pad(win, ((pad_lo, pad_hi), (0, 0)))
        w = win.shape[0]

        out = np.empty((b - a, self.n_cols), dtype=np.float32)
        for r in range(self.base.shape[1]):
            block = fftconvolve(
                win[:, r][:, None], self.B, mode="full", axes=0
            )[self.n_pre:self.n_pre + w]
            out[:, r * self.n_basis:(r + 1) * self.n_basis] = block[self.n_post:self.n_post + (b - a)]
        return out

    def iter_chunks(self, chunk_samples: int = 250 * 300):
        """Yield ``(a, b, X)`` interior chunks tiling the recording.

        Parameters
        ----------
        chunk_samples : int, default 75000
            Interior length per chunk (~300 s at 250 Hz).

        Yields
        ------
        a, b : int
            Interior sample range.
        X : ndarray, shape (b - a, n_cols)
            Expanded design for the range.
        """
        for a in range(0, self.n_samples, chunk_samples):
            b = min(a + chunk_samples, self.n_samples)
            yield a, b, self.expand_range(a, b)


# --- construction ----------------------------------------------------------
def _event_impulses(times: np.ndarray, weights: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Place ``weights`` at the grid samples nearest ``times`` (finite events only)."""
    out = np.zeros(tvec.size, dtype=np.float64)
    ok = np.isfinite(times) & np.isfinite(weights)
    idx = np.searchsorted(tvec, times[ok])
    idx = np.clip(idx, 0, tvec.size - 1)
    np.add.at(out, idx, weights[ok])
    return out


def build_base_regressors(
    trials: pd.DataFrame, cont: io.ContinuousTraces, tvec: np.ndarray
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Assemble base regressors on the LFP grid.

    Event groups carry an unmodulated onset plus a covariate-weighted column
    (contrast / choice / feedback sign); continuous groups carry z-scored
    behaviour traces. Column means are left in place -- ``solve.py`` centres X
    and Y jointly via accumulated moments.

    Returns
    -------
    base : ndarray, shape (n_samples, n_base), float32
    names : list of str
    groups : dict[str, list[int]]
        Group name -> base-column indices.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[int]] = {}

    def add(group: str, name: str, series: np.ndarray) -> None:
        groups.setdefault(group, []).append(len(cols))
        cols.append(series)
        names.append(name)

    ones = np.ones(len(trials))
    # events: onset + signed covariate modulation
    add("stimOn", "stimOn_on", _event_impulses(trials["stimOn_times"].to_numpy(), ones, tvec))
    add("stimOn", "stimOn_contrast",
        _event_impulses(trials["stimOn_times"].to_numpy(), trials["contrast"].to_numpy(), tvec))
    add("move", "move_on", _event_impulses(trials["firstMovement_times"].to_numpy(), ones, tvec))
    add("move", "move_choice",
        _event_impulses(trials["firstMovement_times"].to_numpy(), trials["choice"].to_numpy(), tvec))
    add("feedback", "feedback_on", _event_impulses(trials["feedback_times"].to_numpy(), ones, tvec))
    add("feedback", "feedback_type",
        _event_impulses(trials["feedback_times"].to_numpy(), trials["feedbackType"].to_numpy(), tvec))

    # continuous behaviour, resampled to the grid then standardised. Events are
    # the only universal group; wheel and pupil are gated add-ons included only
    # when the session provides them (~16 % lack usable wheel) *and* their own
    # timestamps span most of the recording (MIN_CONTINUOUS_COVERAGE) -- a trace
    # that only covers a slice of the session is mostly np.interp edge-clamping
    # once resampled, which _zscore then amplifies into a non-physiological outlier
    # column (see MIN_CONTINUOUS_COVERAGE docstring) -- so the events core stays
    # comparable across every insertion.
    if cont.has_wheel and _coverage(cont.wheel_t, tvec) >= MIN_CONTINUOUS_COVERAGE:
        wheel_vel = _resample(cont.wheel_t, cont.wheel_velocity, tvec)
        add("wheel", "wheel_vel", _zscore(wheel_vel))
        add("wheel", "wheel_speed", _zscore(np.abs(wheel_vel)))
    if cont.has_pupil and _coverage(cont.cam_t, tvec) >= MIN_CONTINUOUS_COVERAGE:
        add("pupil", "pupil", _zscore(_resample(cont.cam_t, cont.pupil_diameter, tvec)))

    return np.column_stack(cols).astype(np.float32), names, groups


def make_design(
    pid: str,
    t_pre: float = 1.5,
    t_post: float = 1.5,
    n_basis: int | None = 10,
) -> Design:
    """Build the full lagged design for one insertion.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    t_pre, t_post : float
        Acausal / causal lag half-windows (seconds).
    n_basis : int or None, default 10
        Raised-cosine bump count, or ``None`` for full FIR.

    Returns
    -------
    Design
        Streamable design with populated ``col_slices`` / ``col_names``.
    """
    eid = io.pid_to_eid(pid)
    reader = io.open_lfp(pid)
    try:
        tvec = np.asarray(reader.times, dtype=np.float64)
        fs = float(reader.fs)
    finally:
        reader.close()

    trials = io.load_trials(eid)
    cont = io.load_continuous(eid)
    base, names, groups = build_base_regressors(trials, cont, tvec)
    B, taus, n_pre, n_post = raised_cosine_basis(fs, t_pre, t_post, n_basis)

    n_bas = B.shape[1]
    col_slices = {g: slice(idx[0] * n_bas, (idx[-1] + 1) * n_bas) for g, idx in groups.items()}
    col_names = [f"{nm}@b{k}" for nm in names for k in range(n_bas)]

    return Design(
        pid=pid, eid=eid, fs=fs, tvec=tvec, base=base, base_names=names,
        groups=groups, B=B, taus=taus, n_pre=n_pre, n_post=n_post,
        col_slices=col_slices, col_names=col_names,
    )