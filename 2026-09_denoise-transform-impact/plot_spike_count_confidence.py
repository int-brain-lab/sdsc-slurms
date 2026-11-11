"""Plot waveform-feature reliability against spike count, to pick a NaN-masking
confidence threshold for `EphysDenoiser` (see discussion: low `spike_count`
channels have noisy waveform-derived features, e.g. `polarity`, since they
average over few spikes).

Uses the raw (pre-denoise, pre-log2-transform) `raw_ephys_features.pqt` table, so
`spike_count` is the genuine per-channel spike rate (spikes/s) driving estimation
noise, not the already TV-smoothed value in `raw_ephys_features_denoised.pqt`.
Restricted to good channels (`channel_labels == 0`); the 236 channels with
`spike_count == 0` have no waveform features at all (NaN by construction) and are
dropped.
"""

from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import ephysatlas.data

sns.set_theme(context="notebook")

DATA_DIR = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/features/ea_active/2026_W39/agg_full"
)
FIGURES_DIR = Path.home().joinpath("Documents", "figures")
DATE_PREFIX = "2026-09-22"
N_SAMPLE = 20_000
RANDOM_STATE = 12345

# (column, label, unit)
WAVEFORM_FEATURES = [
    ("polarity", "polarity", "dimensionless"),
    ("alpha_mean", "alpha_mean", "N/A (DARTsort units)"),
    ("alpha_std", "alpha_std", "N/A (DARTsort units)"),
    ("peak_val", "peak_val", "V"),
    ("spatial_spread_um", "spatial_spread_um", "um"),
    ("slowness_s_per_m", "slowness_s_per_m", "s/m"),
]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_DIR.joinpath("raw_ephys_features.pqt"))
    df = df[(df["channel_labels"] == 0) & (df["spike_count"] > 0)]
    print(f"good channels with spikes: {len(df)}")
    # alpha_mean/alpha_std carry a handful of extreme DARTsort-localisation outliers that dwarf
    # the rest of the distribution; the production pipeline clips these the same way (see
    # ephysatlas.aggregation.denoise_raw_features_data) before ever plotting or training on them.
    df = ephysatlas.data.outlier_treatment(df, columns=["alpha_mean", "alpha_std"])

    # Plain distribution of spike counts (real units: spikes/s)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="spike_count", log_scale=True, ax=ax)
    ax.set_xlabel("spike count (spikes/s)")
    ax.set_title("Distribution of per-channel spike count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR.joinpath(f"{DATE_PREFIX}_spike_count_distribution.png"), dpi=150)

    # 2D KDE of each waveform feature against spike count
    df_sample = df.sample(n=min(N_SAMPLE, len(df)), random_state=RANDOM_STATE)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (column, label, unit) in zip(axs.flat, WAVEFORM_FEATURES):
        sns.kdeplot(
            data=df_sample, x="spike_count", y=column, fill=True, log_scale=(True, False), ax=ax
        )
        # view window only (data passed to kdeplot is untouched): alpha_mean/alpha_std are
        # right-skewed enough that a handful of channels stretch the axis and flatten the rest
        ax.set_ylim(df[column].quantile(0.005), df[column].quantile(0.995))
        ax.set_xlabel("spike count (spikes/s)")
        ax.set_ylabel(f"{label} ({unit})")
    fig.suptitle(f"Waveform features vs spike count (n={len(df_sample)} sampled channels)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR.joinpath(f"{DATE_PREFIX}_waveform_features_vs_spike_count.png"), dpi=150)


if __name__ == "__main__":
    main()
