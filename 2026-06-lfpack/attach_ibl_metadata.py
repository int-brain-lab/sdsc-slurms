"""
Attach IBL metadata to lfpack HDF5 archives — single local post-processing step.

Runs on a local machine (not the cluster) against the aggregated archives synced from
ceph.  Consolidates two previously separate passes:

1. **Channels** — brain-location annotations per channel, written to ``<pid>/00/meta``:
   ``ml`` / ``ap`` / ``dv`` (CCF, metres), ``atlas_id`` (int32), ``acronym`` (list[str]).
   Source: the ephys-atlas features dataframe (one bulk load, no per-PID network calls),
   falling back to ``SpikeSortingLoader.load_channels`` for PIDs absent from it.

2. **Sync** — the registered sample↔time knot pairs (as fitted by IBL's
   ``sync_probe_front_times()``, either an exact multi-pulse fit or a smooth 2-point fit,
   whichever that extractor's own defect detection chose) written to every scale's meta via
   ``lfpack.write_sync_attrs``. No re-fitting or residual gating happens here — this script
   trusts the extractor's fit-type choice.

**Position-based join.**  Channel annotations are placed onto the archive's channels by
**exact electrode position**, not by row order: the source coords (``lateral_um`` /
``axial_um``, µm) and the archive geometry (``geometry_x`` / ``geometry_y``) describe the same
electrodes but may use a different coordinate origin and channel order, and the source may be
missing channels (rows dropped from the features parquet).  ``join_channels_to_archive``
reconciles the origin with one integer translation and matches each archive channel to its
source electrode exactly; source-dropped channels are filled with void.  If almost nothing
matches the source is the wrong probe — that PID is reported and channels are skipped (sync is
still attached, as it does not depend on channel order).

Metadata is written into every ``*compressed*.h5`` found under ``--local-root`` (default and
aggressive, BWM and full-atlas), for whichever recordings each archive contains.
"""

import argparse
import logging
from pathlib import Path

import ephysatlas.anatomy
import ephysatlas.data
import h5py
import lfpack
import numpy as np
import tqdm
from brainbox.io.one import SpikeSortingLoader
from lfpack import LFPackReader
from one.alf.exceptions import ALFObjectNotFound
from one.api import ONE

# ── config ───────────────────────────────────────────────────────────────────
LOCAL_ROOT = Path("/Users/olivier/Documents/datadisk/lfp-processing/lfpack")
FEATURES_ROOT = Path.home().joinpath("data", "ephys-atlas", "features")
PROJECT = "ea_active"
# LFP base rate = AP rate / (AP_RESAMPLE × LFP_RESAMPLE); sync pulses are in AP samples.
LFP_AP_RESAMPLE_FACTOR = 12
LFP_RESAMPLE_FACTOR = 10
GEOM_MIN_MATCH = 0.5        # below this exact-match fraction the source is the wrong probe (skip);
                            # above it, unmatched archive channels are just dropped-source channels

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class SyncInvalid(ValueError):
    """Raised when a probe's registered sync knots fail basic structural validation."""


def load_features_by_pid(one):
    """Load the ephys-atlas channel features dataframe, grouped by PID for O(1) lookup."""
    vintage = ephysatlas.data.get_latest_label(one=one, project=PROJECT)
    path_features = FEATURES_ROOT.joinpath(PROJECT, vintage, "agg_full")
    if not path_features.exists():
        ephysatlas.data.download_tables(FEATURES_ROOT, label=vintage, one=one)
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df = ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas, strict=False)
    by_pid = {pid: grp.sort_index() for pid, grp in df.groupby("pid")}
    log.info(f"Loaded {len(by_pid)} PIDs from ephys-atlas features {vintage}")
    return by_pid


def get_channels(pid, features_by_pid, one):
    """Return per-channel annotation for *pid*, or None if unavailable.

    Prefers the features dataframe; falls back to ``SpikeSortingLoader.load_channels``.
    Returns a dict with ``ml``/``ap``/``dv`` (m), ``atlas_id``, ``acronym`` and the on-probe
    ``lateral_um``/``axial_um`` (µm, None if the source does not carry them).
    """
    def _probe_coords(ch, keys):
        lat = ch[keys[0]].to_numpy() if hasattr(ch, "columns") else ch.get(keys[0])
        ax = ch[keys[1]].to_numpy() if hasattr(ch, "columns") else ch.get(keys[1])
        return (None if lat is None else np.asarray(lat, np.float32),
                None if ax is None else np.asarray(ax, np.float32))

    if pid in features_by_pid:
        ch = features_by_pid[pid]
        lateral_um, axial_um = (
            _probe_coords(ch, ("lateral_um", "axial_um"))
            if {"lateral_um", "axial_um"}.issubset(ch.columns) else (None, None)
        )
        return dict(
            ml=ch["x"].to_numpy(np.float32), ap=ch["y"].to_numpy(np.float32),
            dv=ch["z"].to_numpy(np.float32), atlas_id=ch["atlas_id"].to_numpy(np.int32),
            acronym=ch["acronym"].tolist(), lateral_um=lateral_um, axial_um=axial_um,
        )
    try:
        ch = SpikeSortingLoader(one=one, pid=pid).load_channels()
    except Exception as e:  # noqa: BLE001
        log.warning(f"  {pid}: channels FAIL — {e}")
        return None
    return dict(
        ml=ch["x"].astype(np.float32), ap=ch["y"].astype(np.float32),
        dv=ch["z"].astype(np.float32), atlas_id=ch["atlas_id"].astype(np.int32),
        acronym=list(ch["acronym"]),
        lateral_um=ch.get("lateral_um"), axial_um=ch.get("axial_um"),
    )


def compute_sync(ssl):
    """Load the probe's registered sample↔time sync knots, at the LFP base rate.

    ``ssl.samples2times(0)`` triggers the sync download and lets the extractor's own,
    more sophisticated defect detection pick the fit type (``'exact'`` multi-pulse or
    ``'smooth'`` 2-point) — this function trusts that choice and returns the resulting
    knot pairs unchanged, after only a structural sanity check. No affine re-fit and no
    residual/magnitude gating: those would either discard real non-linear structure or
    re-second-guess a decision the extractor already made better.

    Parameters
    ----------
    ssl : brainbox.io.one.SpikeSortingLoader

    Returns
    -------
    samples, times : np.ndarray, np.ndarray
        Sync knots as (sample @ LFP base rate, time in seconds), strictly increasing,
        size >= 2.

    Raises
    ------
    SyncInvalid
        The registered knots are too few, non-monotonic, or non-finite.
    one.alf.exceptions.ALFObjectNotFound
        Propagates unchanged — no sync dataset is registered for this probe.
    """
    ssl.samples2times(0)  # triggers sync download; populates ssl._sync
    fs_ratio = LFP_RESAMPLE_FACTOR * LFP_AP_RESAMPLE_FACTOR
    timestamps = ssl._sync["timestamps"]
    samples = timestamps[:, 0] / fs_ratio  # AP samples → LFP base samples
    times = timestamps[:, 1]               # seconds

    if samples.size < 2:
        raise SyncInvalid(f"fewer than 2 sync knots ({samples.size})")
    if not (np.all(np.isfinite(samples)) and np.all(np.isfinite(times))):
        raise SyncInvalid("sync knots contain non-finite values")
    if not (np.all(np.diff(samples) > 0) and np.all(np.diff(times) > 0)):
        raise SyncInvalid("sync knots are not strictly increasing")
    return samples, times


def join_channels_to_archive(meta, channels):
    """Map source channel annotations onto the archive's channel order by electrode position.

    The archive geometry (``geometry_from_meta``, Reader-sorted order) and the source coords
    (features / SpikeSortingLoader) describe the same electrodes but may use a different
    coordinate origin and channel order, and the source may be *missing channels* (rows
    dropped from the features parquet).  So annotations cannot be copied by row position.

    Electrode positions lie on an exact integer-µm lattice, so we reconcile the origin with a
    single integer translation (per-axis median difference) and then match archive→source by
    **exact position** (a hash lookup, no distance tolerance).  A high exact-match rate itself
    proves the translation is right.  Unmatched archive channels (source-dropped) are filled
    with void (``atlas_id=0``, ``acronym="void"``, ``ml/ap/dv=nan``).

    Returns
    -------
    (annot, detail) : (dict or None, str)
        ``annot`` holds ``ml``/``ap``/``dv``/``atlas_id``/``acronym`` arrays in archive-channel
        order, or None when too few channels match (wrong source probe → caller skips).
    """
    gx = np.round(np.asarray(meta.attrs["geometry_x"], float)).astype(int)
    gy = np.round(np.asarray(meta.attrs["geometry_y"], float)).astype(int)
    nc = len(gx)
    sxu, syu = channels.get("lateral_um"), channels.get("axial_um")
    if sxu is None or syu is None:
        return None, "no source probe coords"
    sx = np.round(np.asarray(sxu, float)).astype(int)
    sy = np.round(np.asarray(syu, float)).astype(int)

    # Reconcile the coordinate-origin convention with one integer translation.
    ox = int(round(np.median(gx) - np.median(sx)))
    oy = int(round(np.median(gy) - np.median(sy)))
    src_by_pos = {(int(sx[j] + ox), int(sy[j] + oy)): j for j in range(len(sx))}
    idx = np.array([src_by_pos.get((int(gx[i]), int(gy[i])), -1) for i in range(nc)])
    matched = idx >= 0
    rate = float(matched.mean())
    if rate < GEOM_MIN_MATCH:
        return None, f"only {rate:.0%} channels matched exactly (offset={ox},{oy})"

    isrc = np.where(matched, idx, 0)

    def gather(key, fill, dtype):
        v = np.asarray(channels[key])[isrc]
        return np.where(matched, v, fill).astype(dtype)

    acr = np.asarray(channels["acronym"], dtype=object)[isrc]
    annot = dict(
        ml=gather("ml", np.nan, np.float32),
        ap=gather("ap", np.nan, np.float32),
        dv=gather("dv", np.nan, np.float32),
        atlas_id=gather("atlas_id", 0, np.int32),
        acronym=[str(acr[i]) if matched[i] else "void" for i in range(nc)],
    )
    n_un = int((~matched).sum())
    return annot, f"exact-matched {rate:.0%}" + (f", {n_un} source-dropped→void" if n_un else "")


def main():
    parser = argparse.ArgumentParser(description="Attach IBL channels + sync metadata to lfpack archives.")
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT,
                        help="Root holding the synced *compressed*.h5 archives.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write attrs.")
    args = parser.parse_args()

    one = ONE()
    features_by_pid = load_features_by_pid(one)

    h5files = sorted(args.local_root.rglob("*compressed*.h5"))
    if not h5files:
        raise SystemExit(f"No *compressed*.h5 under {args.local_root}")
    # union of recordings across all archives → every recording gets annotated
    pids = sorted({pid for f in h5files for pid in LFPackReader.recordings(f)})
    log.info(f"{len(h5files)} archive(s), {len(pids)} unique recording(s)")

    fail_channels, geom_mismatch = [], []
    fail_sync_missing, fail_sync_invalid, fail_sync_unexpected = [], [], []

    for pid in tqdm.tqdm(pids):
        channels = get_channels(pid, features_by_pid, one)
        if channels is None:
            fail_channels.append(pid)

        # Sync: bucket the outcome into exactly one of three categories. The
        # ALFObjectNotFound/SyncInvalid catches wrap *only* compute_sync (inner try) — the
        # SpikeSortingLoader construction stays under the outer bare-Exception catch alone,
        # so an unrelated loader bug can never be miscategorized as "no sync data".
        samples = times = None
        sync_fail = None
        try:
            ssl = SpikeSortingLoader(one=one, pid=pid)
            try:
                samples, times = compute_sync(ssl)
            except ALFObjectNotFound:
                sync_fail = "missing"
                fail_sync_missing.append(pid)
            except SyncInvalid as e:
                log.warning(f"  {pid}: sync invalid — {e}")
                sync_fail = "invalid"
                fail_sync_invalid.append(pid)
        except Exception:
            log.exception(f"  {pid}: unexpected error computing sync")
            sync_fail = "unexpected"
            fail_sync_unexpected.append(pid)

        for h5file in h5files:
            with h5py.File(h5file, "r" if args.dry_run else "a") as f:
                if pid not in f:
                    continue
                meta0 = f[f"{pid}/00/meta"]
                fs_base = float(meta0.attrs["fs"])

                # ── channels (scale 00): exact position join, drop-tolerant ─────
                if channels is not None:
                    annot, detail = join_channels_to_archive(meta0, channels)
                    if annot is None:
                        log.warning(f"  {pid} [{h5file.name}]: geometry join failed ({detail}) — skipping channels")
                        if pid not in geom_mismatch:
                            geom_mismatch.append(pid)
                    elif not args.dry_run:
                        # Brain-location attrs only.  The bad-channel `labels` attr is
                        # written by the compression step (from detect_bad_channels.py) and
                        # must survive this pass untouched — snapshot it and assert it is
                        # unchanged so a future edit here can never silently clobber it.
                        had_labels = "labels" in meta0.attrs
                        labels_before = meta0.attrs["labels"][:] if had_labels else None
                        meta0.attrs["ml"] = annot["ml"]
                        meta0.attrs["ap"] = annot["ap"]
                        meta0.attrs["dv"] = annot["dv"]
                        meta0.attrs["atlas_id"] = annot["atlas_id"]
                        meta0.attrs["acronym"] = annot["acronym"]
                        if had_labels:
                            assert np.array_equal(meta0.attrs["labels"][:], labels_before), (
                                f"{pid} [{h5file.name}]: bad-channel labels changed while "
                                "attaching brain regions"
                            )
                        else:
                            log.warning(f"  {pid} [{h5file.name}]: no bad-channel `labels` "
                                        "attr (was detection fed through at compression?)")

                # ── sync (all scales, rate-scaled from base) ────────────────────
                # Every outcome (success or any of the three failure buckets) must be
                # reflected in every scale's meta on *this* run — otherwise a re-run that
                # fails today would silently keep a previous run's stale sync attrs.
                if not args.dry_run:
                    for key in sorted(k for k in f[pid].keys() if k.isdigit()):
                        meta = f[f"{pid}/{key}/meta"]
                        if sync_fail is None:
                            scale_ratio = float(meta.attrs["fs"]) / fs_base
                            lfpack.write_sync_attrs(meta, samples * scale_ratio, times)
                        else:
                            lfpack.clear_sync_attrs(meta)

    log.info("Done.")
    log.info(f"channels source failures : {len(fail_channels)}  {fail_channels or ''}")
    log.info(f"sync missing (no dataset): {len(fail_sync_missing)}  {fail_sync_missing or ''}")
    log.info(f"sync invalid (bad knots) : {len(fail_sync_invalid)}  {fail_sync_invalid or ''}")
    log.info(f"geometry mismatches      : {len(geom_mismatch)}  {geom_mismatch or ''}")
    log.info(f"sync unexpected failures : {len(fail_sync_unexpected)}  {fail_sync_unexpected or ''}")
    if fail_sync_unexpected:
        log.error(f"{len(fail_sync_unexpected)} PID(s) hit an unexpected sync failure — failing loudly.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()