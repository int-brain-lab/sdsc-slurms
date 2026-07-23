"""
One-off: correct the stored probe geometry in existing lfpack archives.

Early archives stored NP1 geometry (``neuropixel.trace_header(version=1)[:nc]``) for every
recording, so non-NP1 probes (NP2 single-shank, nc=96) carry wrong ``geometry_x`` /
``geometry_y``.  The real geometry is fully determined by the SpikeGLX channel map, which is
already embedded in each archive as the ``sglx_meta`` JSON attr — so we can recompute and
overwrite the two attrs in place, no recompression and no network required.

This is the cheap alternative to recompressing (which reuses the Cadzow checkpoint but still
costs ~5 h on two nodes).  It reproduces exactly what the fixed encoder would store:
``spikeglx.geometry_from_meta(meta, nc=nc, sort=True)`` — ``sort=True`` matches the Reader's
default channel order used by the spatial pre-processing.  NP1 recordings whose meta has no
geometry map are left untouched (their NP1 default is already correct).

Usage
-----
python fix_geometry.py <path> [--dry-run]      # <path> is a file or a directory (rglob)
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import spikeglx


def patch_meta(meta, dry_run):
    """Recompute geometry for one ``…/meta`` group and overwrite in place.

    Returns
    -------
    str
        One of ``"changed"``, ``"unchanged"``, ``"no-map"``, ``"skip"`` (with a reason).
    float
        Max |Δ| (µm) between old and new geometry, NaN when not applicable.
    """
    if "geometry_x" not in meta.attrs or "sglx_meta" not in meta.attrs:
        return "skip", np.nan
    nc = int(meta.attrs["nc"])
    m = json.loads(meta.attrs["sglx_meta"])
    if not m:
        return "skip", np.nan
    th = spikeglx.geometry_from_meta(m, nc=nc, sort=True)
    if th is None:  # no snsGeomMap/snsShankMap → NP1 default already stored, leave as is
        return "no-map", np.nan
    x = np.asarray(th["x"], np.float32)[:nc]
    y = np.asarray(th["y"], np.float32)[:nc]
    if x.size != nc or y.size != nc:
        return "skip", np.nan
    ox = np.asarray(meta.attrs["geometry_x"], np.float32)
    oy = np.asarray(meta.attrs["geometry_y"], np.float32)
    dmax = float(np.max(np.abs(np.r_[x - ox, y - oy])))
    if dmax == 0.0:
        return "unchanged", 0.0
    if not dry_run:
        meta.attrs["geometry_x"] = x
        meta.attrs["geometry_y"] = y
    return "changed", dmax


def process_file(h5file, dry_run):
    """Patch every ``<rec>/<scale>/meta`` in one archive; return a per-status counter."""
    counts = {"changed": 0, "unchanged": 0, "no-map": 0, "skip": 0}
    worst = 0.0
    mode = "r" if dry_run else "a"
    with h5py.File(h5file, mode) as f:
        for rec in f.keys():
            grp = f[rec]
            if not isinstance(grp, h5py.Group):
                continue
            for scale in [k for k in grp.keys() if k.isdigit()]:
                status, dmax = patch_meta(grp[f"{scale}/meta"], dry_run)
                counts[status] += 1
                if status == "changed":
                    worst = max(worst, dmax)
    verb = "would change" if dry_run else "changed"
    print(f"  {h5file.name}: {verb} {counts['changed']}  unchanged {counts['unchanged']}  "
          f"no-map(NP1) {counts['no-map']}  skip {counts['skip']}  max|Δ|={worst:.1f} µm")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Correct probe geometry in lfpack archives.")
    parser.add_argument("path", type=Path, help="Archive file or directory (searched recursively).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write.")
    args = parser.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("lf_compressed*.h5"))
    if not files:
        raise SystemExit(f"No lf_compressed*.h5 under {args.path}")
    print(f"{'DRY-RUN ' if args.dry_run else ''}Patching geometry in {len(files)} archive(s):")

    total = {"changed": 0, "unchanged": 0, "no-map": 0, "skip": 0}
    for h5file in files:
        for k, v in process_file(h5file, args.dry_run).items():
            total[k] += v
    print(f"\nTotal meta groups — changed {total['changed']}  unchanged {total['unchanged']}  "
          f"no-map(NP1) {total['no-map']}  skip {total['skip']}")


if __name__ == "__main__":
    main()
