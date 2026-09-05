"""Region-decoding accuracy report — run locally after rsyncing results back.

Loads every combo's ``accuracy.json`` (written by ``train_classifier.py``) and
produces two separate figures plus a recap table image, kept as two distinct
questions rather than one combined plot:

1. **Compression impact**: accuracy by LFP source (default / aggressive /
   uncompressed), snippet-mode "all" only — isolates the compression-tier
   effect from snippet selection.
2. **Saturation impact**: accuracy by snippet-mode (all vs
   saturation-avoided), one group per LFP source, with fold-to-fold
   variability shown (individual 5-fold points + mean +/- SD) since the
   naive-vs-saturation-avoided gap is small relative to fold noise.

Usage:
    python report.py --results-root /Users/olivier/Documents/datadisk/lfp-processing/lfp-decoders
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook")

# Ordered by increasing compression (decreasing fidelity) for the compression-impact
# plot and recap table. "mild" only has a snippet_mode="all" run (no saturation-avoided
# variant), so the saturation-impact plot uses SATURATION_SOURCE_ORDER instead, which
# excludes it.
SOURCE_ORDER = ["uncompressed", "mild", "default", "aggressive"]
SATURATION_SOURCE_ORDER = ["default", "aggressive", "uncompressed"]
MODE_ORDER = ["all", "saturation-avoided"]
SOURCE_LABELS = {
    "default": "default\n(lfpack ε=150, α=28)",
    "aggressive": "aggressive\n(lfpack ε=450, α=96)",
    "uncompressed": "uncompressed\n(Cadzow reference)",
    "mild": "mild\n(lfpack ε=100, α=14)",
}
FIGURE_DIR = Path.home().joinpath("Documents", "figures")


def load_accuracies(results_root: Path) -> pd.DataFrame:
    """Read every ``<source>/<mode>/accuracy.json`` under ``results_root``."""
    rows = []
    for path in results_root.glob("*/*/accuracy.json"):
        rows.append(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No accuracy.json files found under {results_root}")
    df = pd.DataFrame(rows)
    df["lfp_source"] = pd.Categorical(df["lfp_source"], categories=SOURCE_ORDER, ordered=True)
    df["snippet_mode"] = pd.Categorical(df["snippet_mode"], categories=MODE_ORDER, ordered=True)
    return df.sort_values(["lfp_source", "snippet_mode"])


def explode_folds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (lfp_source, snippet_mode, fold) with that fold's accuracy."""
    rows = [
        {"lfp_source": row.lfp_source, "snippet_mode": row.snippet_mode,
         "fold": i, "fold_accuracy": acc}
        for row in df.itertuples()
        for i, acc in enumerate(row.fold_accuracies)
    ]
    return pd.DataFrame(rows)


def _annotate_bars(ax) -> None:
    for container in ax.containers:
        if hasattr(container, "datavalues"):  # bar containers only, not error bars/points
            ax.bar_label(container, fmt="%.2f", padding=2)


def plot_compression_impact(df: pd.DataFrame, out_path: Path) -> None:
    """Idea 1: accuracy by LFP source only, snippet_mode == 'all'."""
    data = df[df["snippet_mode"] == "all"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=data, x="lfp_source", y="accuracy", order=SOURCE_ORDER,
        hue="lfp_source", palette=sns.color_palette("colorblind", n_colors=len(SOURCE_ORDER)),
        legend=False, ax=ax,
    )
    _annotate_bars(ax)
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCE_ORDER])
    ax.set_ylim(0, 1)
    ax.set_xlabel("LFP source")
    ax.set_ylabel("Region-decoding accuracy (Cosmos, held-out)")
    ax.set_title("Impact of LFP compression tier on region decoding")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_saturation_impact(df: pd.DataFrame, out_path: Path) -> None:
    """Idea 2: accuracy by snippet_mode, grouped by LFP source, with fold-to-fold spread."""
    folds = explode_folds(df)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    palette = sns.color_palette("colorblind", n_colors=len(MODE_ORDER))
    sns.barplot(
        data=folds, x="lfp_source", y="fold_accuracy", hue="snippet_mode",
        order=SATURATION_SOURCE_ORDER, hue_order=MODE_ORDER, palette=palette,
        errorbar="sd", capsize=0.1, err_kws={"linewidth": 1.5, "color": "0.2"},
        ax=ax,
    )
    sns.stripplot(
        data=folds, x="lfp_source", y="fold_accuracy", hue="snippet_mode",
        order=SATURATION_SOURCE_ORDER, hue_order=MODE_ORDER, dodge=True,
        palette=["#262626"] * len(MODE_ORDER), size=4, alpha=0.6, jitter=0.08,
        legend=False, ax=ax,
    )
    ax.set_xticks(range(len(SATURATION_SOURCE_ORDER)))
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SATURATION_SOURCE_ORDER])
    ax.set_ylim(0, 1)
    ax.set_xlabel("LFP source")
    ax.set_ylabel("Region-decoding accuracy (Cosmos, held-out)")
    ax.set_title("Impact of ADC saturation avoidance on region decoding\n(bars: mean ± SD across 5 folds; dots: individual folds)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(MODE_ORDER)], labels[: len(MODE_ORDER)], title="Snippet selection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_recap_table(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Render a recap table (mean accuracy + fold spread + sample sizes) as an image."""
    folds = explode_folds(df)
    stats = folds.groupby(["lfp_source", "snippet_mode"], observed=True)["fold_accuracy"].agg(
        ["mean", "std", "min", "max"]
    )
    table = df.set_index(["lfp_source", "snippet_mode"])[["n_pids", "n_channels"]].join(stats)
    # Not every source has both snippet modes (e.g. "mild" is "all"-only) -- only
    # include combos actually present rather than assuming a full cross-product.
    available = [(s, m) for s in SOURCE_ORDER for m in MODE_ORDER if (s, m) in table.index]
    table = table.loc[available]
    table.index = table.index.set_names(["LFP source", "snippet mode"])
    table = table.rename(columns={
        "mean": "accuracy (mean)", "std": "fold SD", "min": "fold min", "max": "fold max",
        "n_pids": "n PIDs", "n_channels": "n channels",
    })
    display = table.reset_index()
    for col in ["accuracy (mean)", "fold SD", "fold min", "fold max"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.45 * len(display)))
    ax.axis("off")
    mpl_table = ax.table(
        cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center"
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.auto_set_column_width(col=list(range(len(display.columns))))
    mpl_table.scale(1, 1.6)
    for (row, _col), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(sns.color_palette("colorblind")[0])
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()

    df = load_accuracies(args.results_root)
    print(df[["lfp_source", "snippet_mode", "accuracy", "n_channels", "n_pids"]].to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = date.today().isoformat()
    plot_compression_impact(df, FIGURE_DIR.joinpath(f"{prefix}_lfp-decoders_compression-impact.png"))
    plot_saturation_impact(df, FIGURE_DIR.joinpath(f"{prefix}_lfp-decoders_saturation-impact.png"))
    table = plot_recap_table(df, FIGURE_DIR.joinpath(f"{prefix}_lfp-decoders_recap-table.png"))
    print(table.to_string())
    print(f"Figures saved under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
