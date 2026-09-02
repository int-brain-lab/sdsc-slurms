"""Build a saturation-avoiding snippet grid — SDSC cluster driver (or login node).

For goal 2 (impact of ADC saturation on region decoding): for every PID in the
ephys-atlas cohort, reads the **default**-tier archive's saturation intervals
(``LFPackFeatureCalculator.saturation_times()`` -- verified scale-independent,
so this single read covers all three ``--lfp-source`` tiers) and, for each
600 s slot of the usual "default eatools" snippet grid (200 s start, 600 s
spacing -- see ``extract.iter_snippet_t0s``), picks a window in two tiers:

1. **Clear**: the earliest start in the slot whose window overlaps no
   saturated interval at all (``_first_clear_start``).
2. **Minimal-overlap fallback**: when no clear window exists, the start in the
   slot whose window has the *least total* saturated-second overlap
   (``_min_overlap_start``, via an O(log n)-per-query cumulative-saturation
   function). Accepted only if that residual overlap is below
   ``--max-overlap-frac`` of the window (default 50%); otherwise the slot is
   genuinely dropped.

A survey against the real ephys-atlas archive (2026-08-31,
``/Users/olivier/Documents/datadisk/lfp-processing/lfpack/v03/lf_compressed_all.h5``,
1099 PIDs) found the clear-only tier already keeps at least 1 snippet for
every PID (never a full quorum failure) and only drops any slots at all for
3.7% of PIDs; among the 10 worst-hit PIDs, the fallback tier recovers every
dropped slot, with residual overlap typically <2% of the window (worst case
observed: 19%) -- saturation here is heavily fragmented (thousands of
sub-second intervals), not a few long blackouts, so a fallback window is
usually available and cheap.

Writes ``snippets_avoiding_saturation.csv`` (columns: ``pid``, ``t0``,
``i_snippet``, ``saturation_overlap_s`` -- 0.0 for a clear-tier snippet, >0
for a fallback one) for ``extract.py --snippet-mode saturation-avoided`` to
consume (which only needs ``pid``/``t0``; ``saturation_overlap_s`` is for
downstream QC/filtering).

Run once on a login node (cheap: only reads the small saturation table per
PID, not any LF data) before submitting the saturation-avoided extraction
jobs:
    python saturation_snippets.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import extract
from ephysatlas.feature_calculators import LFPackFeatureCalculator

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("saturation_snippets")

MAX_OVERLAP_FRAC = 0.5
FALLBACK_SEARCH_STEP_S = 0.05


def _first_clear_start(
    t0_slot: float, slot_end: float, duration_lf: float, intervals: list[tuple[float, float]]
) -> float | None:
    """Earliest start in ``[t0_slot, slot_end - duration_lf]`` clear of every interval.

    Classic sweep: intervals are checked in start order, each overlap jumps the
    candidate start forward to just past that interval's end. Returns ``None``
    when no clear window remains in the slot.
    """
    candidate = t0_slot
    for start, stop in sorted(intervals):
        if candidate + duration_lf > slot_end:
            return None
        if start < candidate + duration_lf and stop > candidate:
            candidate = stop
    return candidate if candidate + duration_lf <= slot_end else None


def _saturation_cumfunc(intervals: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Breakpoints of the piecewise-linear cumulative-saturated-seconds function.

    ``F(t)`` = total saturated time in ``[0, t)``: rises at rate 1 during a
    saturated interval, flat outside. Any window's saturated overlap is then
    ``F(start + duration) - F(start)`` (via ``np.interp`` against these
    breakpoints) -- exact, since intervals are disjoint, and O(log n) per
    query instead of rescanning every interval per candidate start.
    """
    if not intervals:
        return np.array([0.0, np.inf]), np.array([0.0, 0.0])
    xs, ys = [0.0], [0.0]
    for start, stop in sorted(intervals):
        if start > xs[-1]:
            xs.append(start)
            ys.append(ys[-1])
        xs.append(stop)
        ys.append(ys[-1] + (stop - start))
    xs.append(np.inf)
    ys.append(ys[-1])
    return np.array(xs), np.array(ys)


def _overlap_at(starts: np.ndarray, duration_lf: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.interp(starts + duration_lf, xs, ys) - np.interp(starts, xs, ys)


def _min_overlap_start(
    t0_slot: float,
    slot_end: float,
    duration_lf: float,
    xs: np.ndarray,
    ys: np.ndarray,
    step: float = FALLBACK_SEARCH_STEP_S,
) -> tuple[float, float] | None:
    """(start, overlap_seconds) minimizing saturated overlap in the slot, or None if no room."""
    candidates = np.arange(t0_slot, slot_end - duration_lf + 1e-9, step)
    if candidates.size == 0:
        return None
    overlaps = _overlap_at(candidates, duration_lf, xs, ys)
    i = int(np.argmin(overlaps))
    return float(candidates[i]), float(overlaps[i])


@dataclass
class Snippet:
    t0: float
    saturation_overlap_s: float


def snippet_t0s_avoiding_saturation(
    duration_available: float,
    intervals: list[tuple[float, float]],
    duration_lf: float = extract.DURATION_LF,
    spacing: float = extract.SNIPPET_SPACING_S,
    max_overlap_frac: float = MAX_OVERLAP_FRAC,
) -> tuple[list[Snippet], int]:
    """Return (kept snippets, n_dropped_slots) for one PID."""
    xs = ys = None  # built lazily -- most PIDs/slots never need the fallback tier
    kept: list[Snippet] = []
    dropped = 0
    t0_slot = extract.SNIPPET_T0_START
    while t0_slot + duration_lf <= duration_available:
        slot_end = min(t0_slot + spacing, duration_available)
        start = _first_clear_start(t0_slot, slot_end, duration_lf, intervals)
        if start is not None:
            kept.append(Snippet(t0=start, saturation_overlap_s=0.0))
        else:
            if xs is None:
                xs, ys = _saturation_cumfunc(intervals)
            fallback = _min_overlap_start(t0_slot, slot_end, duration_lf, xs, ys)
            if fallback is not None and fallback[1] <= max_overlap_frac * duration_lf:
                kept.append(Snippet(t0=fallback[0], saturation_overlap_s=fallback[1]))
            else:
                dropped += 1
        t0_slot += spacing
    return kept, dropped


def build_rows_for_pid(pid: str) -> list[dict]:
    calc = LFPackFeatureCalculator(extract.compressed_h5("default"), recording=pid, name=pid)
    _, duration_lf_available = calc.available_duration()
    sat = calc.saturation_times()
    intervals = list(zip(sat["start_time"].to_numpy(), sat["stop_time"].to_numpy()))
    snippets, dropped = snippet_t0s_avoiding_saturation(duration_lf_available, intervals)
    if dropped:
        LOGGER.info("pid=%s: dropped %d/%d slots (saturation)", pid, dropped, dropped + len(snippets))
    n_fallback = sum(1 for s in snippets if s.saturation_overlap_s > 0)
    if n_fallback:
        LOGGER.info("pid=%s: %d/%d snippets used the minimal-overlap fallback", pid, n_fallback, len(snippets))
    return [
        {"pid": pid, "t0": s.t0, "i_snippet": i, "saturation_overlap_s": s.saturation_overlap_s}
        for i, s in enumerate(snippets)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("snippets_avoiding_saturation.csv"),
    )
    args = parser.parse_args()

    pids = extract.available_pids()
    if args.limit is not None:
        pids = pids[: args.limit]

    rows: list[dict] = []
    for pid in pids:
        try:
            rows.extend(build_rows_for_pid(pid))
        except Exception:
            LOGGER.exception("pid=%s: failed to build saturation-avoiding grid", pid)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    n_fallback = int((df["saturation_overlap_s"] > 0).sum()) if len(df) else 0
    LOGGER.info(
        "Wrote %d snippets (%d via minimal-overlap fallback) across %d PIDs -> %s",
        len(df), n_fallback, df["pid"].nunique() if len(df) else 0, args.output,
    )


if __name__ == "__main__":
    main()
