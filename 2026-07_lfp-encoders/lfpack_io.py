"""Data-access layer for the LFP-encoding model.

Thin glue over three sources, all keyed off a single probe insertion id (PID):

- **LFP** decompressed from the BWM lfpack HDF5 archive via
  :class:`lfpack.LFPackReader` (250 Hz, session clock, channel-binned).
- **Trials** table and **continuous behaviour** traces (wheel, DLC pose,
  pupil) from the local ``bwm_behavior`` dataset.
- The **PID -> EID** join from the local ``bwm_ephys`` insertions table.

The ``bwm_behavior`` session shards use ibl-ai-agent's semantic codec, so the
canonical decoder lives in that repo. It is imported lazily and put on
``sys.path`` here (see :func:`_ensure_ibl_ai_agent`) rather than pip-installed,
because ibl-ai-agent is a working tree, not a released package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import types

import numpy as np
import pandas as pd

# --- fixed locations -------------------------------------------------------
LFP_H5 = Path(
    "/Users/olivier/Documents/datadisk/lfp-processing/lfpack/bwm/lf_compressed_all_bwm.h5"
)
IBL_AI_AGENT_ROOT = Path(
    os.environ.get("IBL_AI_AGENT_ROOT", Path.home().joinpath("Documents", "ibl-ai-agent"))
)

# LFP acquisition constants (verified for the BWM lfpack archive).
LFP_FS_NOMINAL = 250.0
BIN_CHANNELS = 4  # adjacent-channel sum -> 96 targets from 384 electrodes


def _ensure_ibl_ai_agent() -> None:
    """Make ``ibl_ai_agent`` importable and satisfy its transitive imports.

    ibl-ai-agent is a source checkout (no wheel), so its repo root is prepended
    to ``sys.path``. Its dataset package imports ``brainwidemap`` at module load
    time only to expose query/download helpers we never call from the decode
    path; ``brainwidemap`` is a GitHub-only package absent from this env, so a
    minimal stub is registered when it is missing. Both actions are idempotent.
    """
    if str(IBL_AI_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(IBL_AI_AGENT_ROOT))
    if "brainwidemap" not in sys.modules:
        try:
            import brainwidemap  # noqa: F401
        except ImportError:
            stub = types.ModuleType("brainwidemap")
            stub.bwm_query = lambda *a, **k: None
            stub.download_aggregate_tables = lambda *a, **k: None
            sys.modules["brainwidemap"] = stub


def resolve_bwm_dir(name: str) -> Path:
    """Return the local root of a BWM dataset (e.g. ``"bwm_behavior"``)."""
    _ensure_ibl_ai_agent()
    from ibl_ai_agent.data_locations import resolve_dataset_dir

    return resolve_dataset_dir(name)


# --- PID <-> EID join ------------------------------------------------------
def pid_to_eid(pid: str) -> str:
    """Resolve a probe insertion id to its session (experiment) id.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.

    Returns
    -------
    str
        Experiment id (EID) on the same clock as trials and LFP.
    """
    ins = pd.read_parquet(resolve_bwm_dir("bwm_ephys").joinpath("metadata", "insertions.parquet"))
    match = ins.loc[ins["pid"] == pid, "eid"]
    if match.empty:
        raise KeyError(f"PID {pid!r} not found in bwm_ephys insertions table")
    return str(match.iloc[0])


# --- LFP -------------------------------------------------------------------
def open_lfp(pid: str, bin_channels: int = BIN_CHANNELS, scale: int = 0):
    """Open a channel-binned LFP reader for one insertion.

    Parameters
    ----------
    pid : str
        Probe insertion UUID (a key of the lfpack archive).
    bin_channels : int, default 4
        Adjacent electrodes summed per target channel.
    scale : int, default 0
        lfpack reconstruction scale (0 = full quality).

    Returns
    -------
    lfpack.LFPackReader
        Reader with ``.times`` on the session clock and ``.read(nsel, csel)``
        returning ``(data, None)``; ``data`` is ``(n_samples, nc)`` float32 volts.
    """
    from lfpack import LFPackReader

    return LFPackReader(str(LFP_H5), recording=pid, scale=scale, bin_channels=bin_channels)


def channels_frame(reader) -> pd.DataFrame:
    """Per-(binned)-channel geometry and anatomy as a tidy frame.

    Columns: ``axial_um`` (depth along the probe), ``lateral_um``, ``x``/``y``/``z``
    (CCF metres), ``atlas_id``, ``acronym``. Row order matches the target axis.
    """
    ch = reader.channels
    return pd.DataFrame({k: np.asarray(v) for k, v in ch.items()})


# --- trials ----------------------------------------------------------------
def load_trials(eid: str) -> pd.DataFrame:
    """Load the trials table for one session, sorted by trial onset.

    Signed contrast is added as ``contrast`` (right positive, left negative;
    the two source columns are mutually exclusive NaNs).
    """
    tr = pd.read_parquet(resolve_bwm_dir("bwm_behavior").joinpath("metadata", "trials.parquet"))
    tr = tr.loc[tr["eid"] == eid].copy()
    if tr.empty:
        raise KeyError(f"EID {eid!r} has no rows in the trials table")
    tr["contrast"] = tr["contrastRight"].fillna(0.0) - tr["contrastLeft"].fillna(0.0)
    return tr.sort_values("intervals_0").reset_index(drop=True)


# --- continuous behaviour --------------------------------------------------
# Minimum fraction of finite pupil frames for the (gated) pupil group to be
# included; below this the trace is mostly interpolation and is dropped.
PUPIL_FINITE_MIN = 0.5


@dataclass
class ContinuousTraces:
    """Raw continuous behaviour traces on their native clocks.

    Each ``*_t`` array is a session-clock time base; signals share the base
    named in their prefix. Camera signals are DLC left-camera outputs (~60 Hz);
    wheel is ~1 kHz. All resampling onto the LFP grid happens in ``design.py``.

    The ``bwm_behavior`` v1.1.0 shards do not carry every stream on every
    session (~16 % lack usable wheel, some lack the left camera), so loading is
    total: ``has_wheel`` / ``has_pupil`` flag which optional groups are usable
    and their arrays are empty when the flag is ``False``. Events are the only
    universal group; a PID missing wheel/pupil is still fitted with what it has
    rather than dropped.
    """

    wheel_t: np.ndarray
    wheel_velocity: np.ndarray  # signed, rad/s
    wheel_position: np.ndarray
    cam_t: np.ndarray
    pupil_diameter: np.ndarray  # smoothed DLC pupil diameter (px)
    has_wheel: bool
    has_pupil: bool


def _col(features: np.ndarray, columns: list[str], suffix: str) -> np.ndarray | None:
    """Return the DLC feature column whose name ends with ``suffix``, or ``None``."""
    idx = next((i for i, c in enumerate(columns) if c.endswith(suffix)), None)
    return None if idx is None else features[:, idx]


def load_continuous(eid: str) -> ContinuousTraces:
    """Decode the behaviour session shard into continuous traces.

    Never raises on a missing stream: sessions lacking wheel velocity or the
    left camera return empty arrays with the corresponding flag set ``False``.

    Parameters
    ----------
    eid : str
        Experiment id.

    Returns
    -------
    ContinuousTraces
        Wheel velocity/position and left-camera pupil diameter on their native
        clocks (session time in seconds), with per-stream availability flags.
    """
    _ensure_ibl_ai_agent()
    from ibl_ai_agent.datasets.bwm_behavior_compression import read_behavior_session_shard

    shard = read_behavior_session_shard(
        resolve_bwm_dir("bwm_behavior").joinpath("sessions", f"{eid}.zip")
    )
    arr = shard["arrays"]
    empty = np.empty(0, dtype=np.float64)

    # wheel: both timestamps and velocity must be present to be usable
    has_wheel = "wheel.timestamps" in arr and "wheel.velocity" in arr
    wheel_t = np.asarray(arr["wheel.timestamps"], dtype=np.float64) if has_wheel else empty
    wheel_v = np.asarray(arr["wheel.velocity"], dtype=np.float64) if has_wheel else empty
    wheel_p = np.asarray(arr.get("wheel.position", empty), dtype=np.float64)

    # pupil: needs the left-camera feature array, the smoothed-pupil column, and
    # enough finite frames that the resampled trace is not mostly interpolation
    cam_t, pupil, has_pupil = empty, empty, False
    cams = shard["meta"].get("cameras", {})
    if "leftCamera.features" in arr and "leftCamera" in cams:
        col = _col(np.asarray(arr["leftCamera.features"]),
                   cams["leftCamera"]["columns"], "pupilDiameter_smooth")
        if col is not None:
            pupil = col.astype(np.float64)
            cam_t = np.asarray(arr["leftCamera.timestamps"], dtype=np.float64)
            has_pupil = bool(np.isfinite(pupil).mean() >= PUPIL_FINITE_MIN)

    return ContinuousTraces(
        wheel_t=wheel_t, wheel_velocity=wheel_v, wheel_position=wheel_p,
        cam_t=cam_t, pupil_diameter=pupil, has_wheel=has_wheel, has_pupil=has_pupil,
    )