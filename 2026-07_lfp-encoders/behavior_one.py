"""ONE-backed behaviour loaders for the brain-wide LFP-encoding cluster run.

These replace the ``bwm_behavior``-shard loaders used on the laptop
(``lfpack_io.load_trials`` / ``load_continuous``). Loading straight from ONE via
:class:`brainbox.io.one.SessionLoader` sidesteps the compressed-dataset wheel gap
(int-brain-lab/ibl-ai-agent#18): wheel is resolved from the raw ALF (revision-aware)
and returned already interpolated to a uniform grid with a filtered velocity.

Returned objects match what ``design.build_base_regressors`` expects:
- ``load_trials_one`` → a trials DataFrame with a signed ``contrast`` column.
- ``load_continuous_one`` → an ``lfpack_io.ContinuousTraces`` with ``has_wheel`` /
  ``has_pupil`` gating flags (empty arrays when a stream is absent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader

from lfpack_io import PUPIL_FINITE_MIN, ContinuousTraces

WHEEL_FS = 100.0  # uniform wheel sampling rate (Hz); see issue #18


def load_trials_one(eid: str, one) -> pd.DataFrame:
    """Load the trials table for one session and add a signed contrast column.

    Parameters
    ----------
    eid : str
        Experiment id.
    one : ONE
        An initialised ONE / OneSdsc instance.

    Returns
    -------
    pandas.DataFrame
        Trials with ``stimOn_times``, ``firstMovement_times``, ``feedback_times``,
        ``choice``, ``feedbackType`` and a derived ``contrast`` (right − left).
    """
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    tr = sl.trials.copy()
    tr["contrast"] = tr["contrastRight"].fillna(0.0) - tr["contrastLeft"].fillna(0.0)
    return tr


def load_continuous_one(eid: str, one) -> ContinuousTraces:
    """Load wheel and pupil for one session, gated by availability.

    Wheel is interpolated to ``WHEEL_FS`` with a filtered velocity; pupil is the
    smoothed left-camera diameter, included only if enough frames are finite.
    Never raises: a missing/failed stream yields empty arrays and a ``False`` flag.

    Parameters
    ----------
    eid : str
        Experiment id.
    one : ONE
        An initialised ONE / OneSdsc instance.

    Returns
    -------
    lfpack_io.ContinuousTraces
        Wheel + pupil traces with ``has_wheel`` / ``has_pupil`` flags.
    """
    sl = SessionLoader(eid=eid, one=one)
    empty = np.empty(0, dtype=np.float64)

    wheel_t = wheel_v = wheel_p = empty
    has_wheel = False
    try:
        sl.load_wheel(fs=WHEEL_FS)
        wheel_t = sl.wheel["times"].to_numpy(dtype=np.float64)
        wheel_v = sl.wheel["velocity"].to_numpy(dtype=np.float64)
        wheel_p = sl.wheel["position"].to_numpy(dtype=np.float64)
        has_wheel = wheel_t.size > 1
    except Exception:  # noqa: BLE001 - absent/failed wheel degrades to events-only
        pass

    cam_t = pupil = empty
    has_pupil = False
    try:
        # load_pose gives per-frame camera timestamps; load_pupil the smoothed diameter
        sl.load_pose(views=["left"], likelihood_thr=0.9)
        cam_t = sl.pose["leftCamera"]["times"].to_numpy(dtype=np.float64)
        sl.load_pupil(snr_thresh=5.0)
        pupil = sl.pupil["pupilDiameter_smooth"].to_numpy(dtype=np.float64)
        has_pupil = pupil.size == cam_t.size and bool(
            np.isfinite(pupil).mean() >= PUPIL_FINITE_MIN
        )
    except Exception:  # noqa: BLE001 - absent/failed pupil degrades gracefully
        pass

    return ContinuousTraces(
        wheel_t=wheel_t, wheel_velocity=wheel_v, wheel_position=wheel_p,
        cam_t=cam_t, pupil_diameter=pupil, has_wheel=has_wheel, has_pupil=has_pupil,
    )