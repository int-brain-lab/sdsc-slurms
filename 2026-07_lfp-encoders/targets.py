"""LFP targets (Y) for the encoding model.

Two target families share the same design matrix (decision D1):

- **raw broadband voltage** -- the phase-locked evoked potential. Only slow
  drift is removed (per-channel linear detrend); the rest is left to the
  regression.
- **band-power envelopes** -- for each of delta/theta/beta/gamma the signal is
  band-passed, its analytic (Hilbert) amplitude taken, and log-scaled. Each band
  is a separate 96-channel block, so a band-mode ``Y`` has ``4 * 96`` columns.

Unlike the design matrix ``X`` (streamed because it can reach ~100 GB for full
FIR), one probe's LFP is only ~350 MB (96 binned channels x ~917 k samples), so
``Y`` is read once into memory. Band-passing and the Hilbert transform are then
applied to the **whole** trace, which is both simpler and artefact-free at chunk
boundaries; per-channel moments used downstream are therefore exactly global.
``solve.py`` slices this in-memory ``Y`` per time-chunk to pair with ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, hilbert, sosfiltfilt

import lfpack_io as io

# Frequency bands (Hz), capped below the 100 Hz compression fmax / 125 Hz Nyquist.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "beta": (15.0, 30.0),
    "gamma": (30.0, 90.0),
}
_BUTTER_ORDER = 4


def read_full_lfp(
    pid: str, bin_channels: int = io.BIN_CHANNELS, chunk_s: float = 300.0
) -> tuple[np.ndarray, pd.DataFrame, float, np.ndarray]:
    """Decompress a whole recording into memory, channel-binned.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    bin_channels : int, default 4
        Adjacent electrodes summed per target.
    chunk_s : float, default 300.0
        Read granularity (seconds) to bound peak decompression memory.

    Returns
    -------
    volts : ndarray, shape (n_samples, n_channels), float32
        LFP voltage on the session clock.
    channels : DataFrame
        Per-channel geometry/anatomy (row-aligned to ``volts`` columns).
    fs : float
        Sampling rate (Hz).
    tvec : ndarray, shape (n_samples,)
        Session-clock sample times.
    """
    reader = io.open_lfp(pid, bin_channels=bin_channels)
    try:
        n, nc, fs = reader.ns, reader.nc, float(reader.fs)
        volts = np.empty((n, nc), dtype=np.float32)
        step = int(round(chunk_s * fs))
        for a in range(0, n, step):
            b = min(a + step, n)
            volts[a:b], _ = reader.read(slice(a, b), slice(None))
        channels = io.channels_frame(reader)
        tvec = np.asarray(reader.times, dtype=np.float64)
    finally:
        reader.close()
    return volts, channels, fs, tvec


def _band_envelope(volts: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Log analytic-amplitude envelope of one band (zero-phase filtered, global)."""
    sos = butter(_BUTTER_ORDER, band, btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, volts, axis=0), axis=0))
    # floor the envelope a few orders of magnitude below its median before log,
    # so silent channels/samples cannot produce -inf.
    floor = np.median(env, axis=0, keepdims=True) * 1e-6 + np.finfo(np.float32).tiny
    return np.log(np.maximum(env, floor)).astype(np.float32)


@dataclass
class Targets:
    """In-memory LFP targets for one insertion.

    Attributes
    ----------
    pid : str
        Insertion id.
    kind : {"raw", "band"}
        Target family.
    Y : ndarray, shape (n_samples, n_targets), float32
        Targets (raw voltage or stacked log band-power envelopes).
    fs : float
        Sampling rate (Hz).
    tvec : ndarray
        Session-clock sample times (mirrors the design's ``tvec``).
    channels : DataFrame
        Per-(binned)-channel geometry/anatomy.
    target_meta : DataFrame
        One row per target column: ``band`` ("raw" or band name), ``channel``
        (index into ``channels``), plus that channel's ``axial_um``/``acronym``.
    """

    pid: str
    kind: str
    Y: np.ndarray
    fs: float
    tvec: np.ndarray
    channels: pd.DataFrame
    target_meta: pd.DataFrame

    @property
    def n_samples(self) -> int:
        return self.Y.shape[0]

    @property
    def n_targets(self) -> int:
        return self.Y.shape[1]


def _target_meta(channels: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Build the column-metadata frame for one or more stacked channel blocks."""
    nc = len(channels)
    rows = []
    for band in bands:
        for ch in range(nc):
            row = channels.iloc[ch]
            rows.append({
                "band": band,
                "channel": ch,
                "axial_um": float(row["axial_um"]),
                "acronym": str(row["acronym"]),
                "atlas_id": int(row["atlas_id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })
    return pd.DataFrame(rows)


def make_targets(
    pid: str, kind: str = "band", bands: dict[str, tuple[float, float]] | None = None
) -> Targets:
    """Build LFP targets for one insertion.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    kind : {"raw", "band"}, default "band"
        ``"raw"`` returns per-channel linearly-detrended broadband voltage;
        ``"band"`` returns log band-power envelopes stacked over ``bands``.
    bands : dict, optional
        Band name -> (low, high) Hz. Defaults to :data:`BANDS`. Ignored for raw.

    Returns
    -------
    Targets
        In-memory targets with column metadata for depth/region figures.
    """
    volts, channels, fs, tvec = read_full_lfp(pid)
    return targets_from_lfp(pid, kind, volts, channels, fs, tvec, bands=bands)


def targets_from_lfp(
    pid: str,
    kind: str,
    volts: np.ndarray,
    channels: pd.DataFrame,
    fs: float,
    tvec: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
) -> Targets:
    """Build :class:`Targets` from already-read LFP arrays (source-agnostic).

    Factored out of :func:`make_targets` so callers can supply LFP from any source
    -- e.g. the uncompressed Cadzow checkpoint on the cluster -- while reusing the
    raw/band transforms and column metadata. Arguments mirror ``read_full_lfp``'s
    return.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    kind : {"raw", "band"}
        Target family (see :func:`make_targets`).
    volts : ndarray, shape (n_samples, n_channels), float32
        Channel-binned LFP voltage on the session clock.
    channels : DataFrame
        Per-channel geometry/anatomy (row-aligned to ``volts`` columns).
    fs : float
        Sampling rate (Hz).
    tvec : ndarray, shape (n_samples,)
        Session-clock sample times.
    bands : dict, optional
        Band name -> (low, high) Hz. Defaults to :data:`BANDS`. Ignored for raw.

    Returns
    -------
    Targets
        In-memory targets with column metadata for depth/region figures.
    """
    if kind == "raw":
        Y = detrend(volts, axis=0, type="linear").astype(np.float32)
        meta = _target_meta(channels, ["raw"])
    elif kind == "band":
        bands = bands or BANDS
        Y = np.concatenate([_band_envelope(volts, fs, b) for b in bands.values()], axis=1)
        meta = _target_meta(channels, list(bands))
    else:
        raise ValueError(f"kind must be 'raw' or 'band', got {kind!r}")

    return Targets(pid=pid, kind=kind, Y=Y, fs=fs, tvec=tvec, channels=channels, target_meta=meta)
