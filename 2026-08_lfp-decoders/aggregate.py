"""Aggregate extracted snippet features into per-(source, mode) tables.

Thin wrapper around ``ephysatlas.aggregation.produce_output_dataframes`` --
no new aggregation logic: extraction already writes standard OOP outputs
(``channels.pqt`` per PID, per-snippet ``lf_features.pqt``/``csd_features.pqt``,
and a ``snippets_df.pqt`` manifest at the combo root), which that function
already knows how to combine (see the 2026-08-31 plan's "Design grounding").

Run once per (--lfp-source, --snippet-mode) combo, after its extraction job
has finished:
    python aggregate.py --lfp-source default --snippet-mode all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import extract
from ephysatlas.aggregation import produce_output_dataframes

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("aggregate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfp-source", choices=extract.SOURCES, required=True)
    parser.add_argument("--snippet-mode", choices=extract.SNIPPET_MODES, default="all")
    parser.add_argument("--output-root", type=Path, default=extract.OUTPUT_ROOT)
    args = parser.parse_args()

    combo_dir = args.output_root.joinpath(args.lfp_source, args.snippet_mode)
    snippets_df = pd.read_parquet(combo_dir.joinpath("snippets_df.pqt"))
    LOGGER.info(
        "lfp_source=%s snippet_mode=%s: aggregating %d snippets across %d PIDs",
        args.lfp_source, args.snippet_mode, len(snippets_df), snippets_df["pid"].nunique(),
    )

    df_channels, df_raw_ephys, df_features_denoise = produce_output_dataframes(
        snippets_df, input_dir=combo_dir, output_dir=combo_dir
    )
    LOGGER.info(
        "Wrote channels.pqt (%d rows), raw_ephys_features.pqt (%d rows), "
        "raw_ephys_features_denoised.pqt (%d rows) -> %s",
        len(df_channels), len(df_raw_ephys), len(df_features_denoise), combo_dir,
    )


if __name__ == "__main__":
    main()
