"""LFP feature extraction — SDSC cluster driver.

Extracts LF/CSD features (``ephysatlas.feature_calculators``) over the
**ephys-atlas cohort** (every histology-resolved insertion compressed by
``2026-06-lfpack/compress.py`` -- a strict superset of the curated BWM
behavioural cohort used by ``2026-07_lfp-encoders``), once per **LFP source**,
to measure how lossy compression affects region-decoding accuracy:

    --lfp-source default       lfpack SVD+WP (eps=150, alpha=28)   lf_compressed_all.h5
    --lfp-source aggressive    lfpack SVD+WP (eps=450, alpha=96)   lf_compressed_aggressive_all.h5
    --lfp-source uncompressed  Cadzow checkpoint (250 Hz CAR, pre-SVD/WP) <pid>/lf_resampled_car_cadzow.npy

Both compressed tiers read from the **ephys-atlas superset** archive (private
S3: ``resources/lfp/ephys-atlas/``) -- NOT the public ``_bwm`` flagship
archive, which only covers the curated BWM subset. The uncompressed reference
is the per-PID Cadzow checkpoint already written by ``compress.py`` to
``CADZOW_ROOT`` (same tree, no separate download needed); it borrows its
channel geometry/metadata from the default-tier archive (same probe, same
channel layout).

PID list and snippet grid are derived directly from the archive
(``LFPackReader.recordings``, and each PID's own duration), not from the
legacy ``sdsc-slurms/ephys-atlas/snippets_*.csv`` files, which may not match
it exactly. The 600 s-spaced, 200 s-start snippet grid matches the legacy
``ephys_atlas.features`` convention ("the default eatools" snippet
configuration); ``--snippet-mode saturation-avoided`` instead reads
``snippets_avoiding_saturation.csv`` (see ``saturation_snippets.py``).

The archive's LF is already destriped/CAR'd/decimated/Cadzow-denoised
upstream, so every snippet is computed with ``skip_lf_destripe=True`` and
``CsdParams(decimate=1, denoise=False)`` -- see the two new backends'
docstrings (``ephysatlas.feature_calculators.lfpack`` /
``.numpy_array``) for why re-destriping/re-denoising would be wrong (and, for
destriping, outright crash at the decimated rate).
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from ephysatlas.feature_calculators import (
    CsdParams,
    FeatureComputationOptions,
    FeatureParams,
    LFPackFeatureCalculator,
    NumpyArrayFeatureCalculator,
    SnippetWindow,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("extract")

# ── SDSC paths (verified on the cluster 2026-08-31) ──────────────────────────
# The consolidated lfpack archives are downloaded via ephysatlas.data's own
# helper (see download_lfp below) into LFP_DOWNLOAD_ROOT/<project>/lfp_aggregates/
# -- that S3 object (aggregates/atlas/projects/<project>/lfp_aggregates/) is the
# region-annotated archive (attach_ibl_metadata.py's output, confirmed
# byte-identical to the local annotated copy at
# ~/Documents/datadisk/lfp-processing/lfpack/v03/). The raw per-PID archives
# under 2026-06-lfpack/compress.py's own OUTPUT_ROOT
# (/mnt/home/owinter/ceph/ea/denoised_lfp) are NOT annotated -- do not point
# LFP_DATA_ROOT there.
LFP_DOWNLOAD_ROOT = Path("/mnt/home/owinter/ceph/lfp-decoders")
LFP_PROJECT = "ibl_neuropixel_brainwide_01"
LFP_DATA_ROOT = LFP_DOWNLOAD_ROOT.joinpath(LFP_PROJECT, "lfp_aggregates")
# Per-PID Cadzow checkpoints, written by 2026-06-lfpack/compress.py -- already
# on the cluster, no download needed (same tree that job wrote to; carries no
# metadata of its own -- extract.py borrows geometry/channel_metadata for this
# source from the default-tier LFPackFeatureCalculator, so it inherits the
# annotation from LFP_DATA_ROOT once that's downloaded).
CADZOW_ROOT = Path("/mnt/home/owinter/ceph/ea/denoised_lfp")
OUTPUT_ROOT = Path("/mnt/home/owinter/ceph/lfp-decoders/results")

UNCOMPRESSED = "uncompressed"
CADZOW_NPY = "lf_resampled_car_cadzow.npy"
COMPRESSED_FILES = {
    "default": "lf_compressed_all.h5",
    "aggressive": "lf_compressed_aggressive_all.h5",
}
SOURCES = (*COMPRESSED_FILES, UNCOMPRESSED)
SNIPPET_MODES = ("all", "saturation-avoided")

# Same 8.79 s window as the legacy `ephys_atlas.features` pipeline
# (`ns_ap = 2**18` samples at 30 kHz AP; LF duration is the same time span).
DURATION_LF = 2**18 / 30000.0
SNIPPET_T0_START = 200.0
SNIPPET_SPACING_S = 600.0


def compressed_h5(source: str) -> Path:
    """Local path to the consolidated lfpack archive for a compressed ``source``."""
    return LFP_DATA_ROOT.joinpath(COMPRESSED_FILES[source])


def download_lfp(source: str, one=None) -> Path:
    """Fetch the region-annotated, consolidated lfpack archive for ``source``.

    Thin wrapper around ``ephysatlas.data.download_lfp_features`` (the
    canonical S3 accessor for this exact archive -- private bucket, needs
    Alyx-issued AWS credentials via an authenticated ``one.api.ONE``/``OneSdsc``
    instance). ``level`` maps 1:1 to our ``source`` name for the two
    compressed tiers.
    """
    import ephysatlas.data as ea_data

    return ea_data.download_lfp_features(
        LFP_DOWNLOAD_ROOT, project=LFP_PROJECT, one=one, level=source
    )


def available_pids(source: str = "default") -> list[str]:
    """PIDs held in the ephys-atlas-superset archive (the recording universe)."""
    from lfpack import LFPackReader

    return sorted(LFPackReader.recordings(str(compressed_h5(source))))


def iter_snippet_t0s(duration_available: float, duration_lf: float = DURATION_LF):
    """Yield the 600 s-spaced, 200 s-start snippet start times fitting in a duration.

    Matches the legacy ``ephys_atlas.features`` / ``snippets_bwm.csv`` grid
    convention ("the default eatools" snippet configuration).
    """
    t0 = SNIPPET_T0_START
    while t0 + duration_lf <= duration_available:
        yield t0
        t0 += SNIPPET_SPACING_S


def build_calculator(pid: str, source: str):
    """Build the right OOP feature calculator for one PID/source."""
    if source == UNCOMPRESSED:
        default_calc = LFPackFeatureCalculator(
            compressed_h5("default"), recording=pid, name=pid
        )
        geometry = default_calc.load_geometry()
        channel_metadata = default_calc.load_channel_metadata()
        fs_lf = float(default_calc.reader.fs)
        nc = len(np.asarray(geometry["x"]))
        npy_path = CADZOW_ROOT.joinpath(pid, CADZOW_NPY)
        # Zero-copy memmap view: (samples, channels) on disk -> (channels,
        # samples). The backend converts only the small per-snippet slice to
        # float32 -- never materialize/convert the whole (multi-GB) array.
        lf = np.load(npy_path, mmap_mode="r")[:, :nc].T
        return NumpyArrayFeatureCalculator(
            lf=lf,
            fs_lf=fs_lf,
            geometry=geometry,
            name=pid,
            channel_metadata=channel_metadata,
        )
    return LFPackFeatureCalculator(compressed_h5(source), recording=pid, name=pid)


def _snippet_options(output_dir: Path, overwrite: bool) -> FeatureComputationOptions:
    return FeatureComputationOptions(
        features_to_compute=["lf", "csd"],
        skip_lf_destripe=True,
        feature_params=FeatureParams(csd=CsdParams(decimate=1, denoise=False)),
        include_trajectory=False,
        output_dir=output_dir,
        skip_saved_computation=not overwrite,
    )


def process_pid(
    pid: str,
    source: str,
    output_dir: Path,
    snippet_mode: str,
    saturation_t0s: dict[str, list[float]],
    overwrite: bool,
) -> list[dict]:
    """Extract every snippet for one PID; return its manifest records.

    Never raises: a PID/snippet failure is logged and skipped so one bad
    recording does not stop the batch (same per-PID try/except convention as
    ``2026-07_lfp-encoders/encode.py``).
    """
    records: list[dict] = []
    try:
        calc = build_calculator(pid, source)
        if snippet_mode == "saturation-avoided":
            t0s = saturation_t0s.get(pid, [])
        else:
            _, duration_lf = calc.available_duration()
            t0s = list(iter_snippet_t0s(duration_lf))
    except Exception:
        LOGGER.error("pid=%s source=%s: failed to open calculator\n%s", pid, source, traceback.format_exc())
        return records

    for t0 in t0s:
        window = SnippetWindow(t_start=t0, duration_ap=DURATION_LF, duration_lf=DURATION_LF)
        try:
            result = calc.compute_snippet(window, _snippet_options(output_dir, overwrite))
            records.append(dict(result.manifest_record))
        except Exception:
            LOGGER.error(
                "pid=%s source=%s t0=%.1f: snippet failed\n%s",
                pid, source, t0, traceback.format_exc(),
            )
    return records


def _load_saturation_t0s(path: Path) -> dict[str, list[float]]:
    df = pd.read_csv(path)
    return {pid: sorted(group["t0"].tolist()) for pid, group in df.groupby("pid")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfp-source", choices=SOURCES, required=True)
    parser.add_argument("--snippet-mode", choices=SNIPPET_MODES, default="all")
    parser.add_argument("--limit", type=int, default=None, help="process at most N PIDs (smoke runs)")
    parser.add_argument("--workers", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="recompute snippets even if cached")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--download", action="store_true", help="fetch the compressed archive and exit")
    parser.add_argument(
        "--saturation-csv",
        type=Path,
        default=Path(__file__).with_name("snippets_avoiding_saturation.csv"),
        help="CSV built by saturation_snippets.py, used when --snippet-mode saturation-avoided",
    )
    args = parser.parse_args()

    if args.download:
        try:
            from deploy.iblsdsc import OneSdsc as ONE
        except ImportError:
            from one.api import ONE
        one = ONE()
        for source in COMPRESSED_FILES:
            LOGGER.info("Downloading %s ...", COMPRESSED_FILES[source])
            download_lfp(source, one=one)
        return

    output_dir = args.output_root.joinpath(args.lfp_source, args.snippet_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    pids = available_pids()
    if args.limit is not None:
        pids = pids[: args.limit]
    LOGGER.info(
        "lfp_source=%s snippet_mode=%s: %d PIDs -> %s",
        args.lfp_source, args.snippet_mode, len(pids), output_dir,
    )

    saturation_t0s: dict[str, list[float]] = {}
    if args.snippet_mode == "saturation-avoided":
        saturation_t0s = _load_saturation_t0s(args.saturation_csv)

    jobs = (
        joblib.delayed(process_pid)(
            pid,
            args.lfp_source,
            output_dir,
            args.snippet_mode,
            saturation_t0s,
            args.overwrite,
        )
        for pid in pids
    )
    results = joblib.Parallel(n_jobs=args.workers, verbose=10)(jobs)

    records = [record for pid_records in results for record in pid_records]
    snippets_df = pd.DataFrame.from_records(records)
    snippets_df.to_parquet(output_dir.joinpath("snippets_df.pqt"))
    LOGGER.info(
        "Done: %d snippets across %d PIDs -> %s",
        len(snippets_df), snippets_df["pid"].nunique() if len(snippets_df) else 0,
        output_dir.joinpath("snippets_df.pqt"),
    )


if __name__ == "__main__":
    main()
