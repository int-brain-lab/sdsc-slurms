"""
LFP compression QC report — runs locally on the merged archives synced from ceph.

Aggregates, per recording (PID) and per chunk, the three QC axes:

* **saturation** — fraction of the raw recording flagged as ADC-clipped, interval count,
  and whether it was muted (from ``/<pid>/saturation`` attrs);
* **bad channels** — counts of dead / noisy / outside-brain channels (from the
  ``labels`` attr on ``/<pid>/00/meta``, written via detect_bad_channels.py → compress.py);
* **compression** — per-chunk SVD / wavelet-packet / total compression ratios and RMSE
  (from each ``chunks/<i>`` group's attrs).

Reads every ``lf_compressed*all*.h5`` under ``--local-root``.  The ``default`` and
``aggressive`` passes (and ``bwm`` vs full-atlas subsets) are tagged in ``pass`` / ``subset``
columns.  Saturation and bad-channel figures dedupe to one row per PID (they are
pass-independent recording properties copied into both archives).

Outputs
-------
    <out-dir>/lfp_qc_per_pid.pqt     one row per (pid, pass, subset)
    <out-dir>/lfp_qc_per_chunk.pqt   one row per (pid, pass, subset, chunk)
    ~/Documents/figures/2026-07-22_lfp_qc_*.png

Usage
-----
    python qc_report.py                      # default local root, write pqt + figures
    python qc_report.py --no-figures         # tables only
    python qc_report.py --local-root <dir> --out-dir <dir>
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FIG_DATE = "2026-07-22"  # do not change once committed
LOCAL_ROOT = Path("/Users/olivier/Documents/datadisk/lfp-processing/lfpack")
FIG_DIR = Path.home().joinpath("Documents", "figures")
LABEL_NAMES = {0: "good", 1: "dead", 2: "noisy", 3: "outside"}


def _pass_subset(h5file):
    """Derive (pass, subset) tags from a merged-archive filename."""
    name = h5file.name
    return (
        "aggressive" if "aggressive" in name else "default",
        "bwm" if "bwm" in name else "full",
    )


def scan_archive(h5file):
    """Read one merged archive → (per-PID rows, per-chunk rows).

    Parameters
    ----------
    h5file : Path
        A merged ``lf_compressed*all*.h5`` archive.

    Returns
    -------
    tuple(list[dict], list[dict])
        Per-PID summary rows and per-chunk metric rows.
    """
    pass_name, subset = _pass_subset(h5file)
    per_pid, per_chunk = [], []
    with h5py.File(h5file, "r") as f:
        pids = [k for k in f.keys() if f"{k}/00/meta" in f]
        for pid in pids:
            meta = f[f"{pid}/00/meta"].attrs
            nc = int(meta["nc"])
            fs = float(meta["fs"])
            ns_total = int(meta["ns_total"])

            # ── bad channels ────────────────────────────────────────────────
            counts = {f"n_{v}": 0 for v in LABEL_NAMES.values()}
            n_bad = np.nan
            if "labels" in meta:
                labels = np.asarray(meta["labels"]).astype(int)
                for code, name in LABEL_NAMES.items():
                    counts[f"n_{name}"] = int(np.sum(labels == code))
                n_bad = int(np.sum(labels != 0))

            # ── saturation (recording-level, pass-independent) ──────────────
            sat = dict(saturated_fraction=np.nan, n_saturated_intervals=np.nan,
                       total_saturated_sec=np.nan, muted=np.nan)
            skey = f"{pid}/saturation"
            if skey in f:
                sds = f[skey]
                sa = sds.attrs
                fs_raw = float(sa.get("fs", fs))
                n_sat = int(sa.get("n_saturated_samples", 0))
                sat = dict(
                    saturated_fraction=float(sa.get("saturated_fraction", 0.0)),
                    n_saturated_intervals=int(sds.shape[0]) if sds.ndim == 2 else 0,
                    total_saturated_sec=n_sat / fs_raw if fs_raw else 0.0,
                    muted=bool(sa.get("muted", False)),
                )

            # ── compression metrics (per chunk) ─────────────────────────────
            cg = f[f"{pid}/00/chunks"]
            cr_total, cr_svd, cr_wp, rmse = [], [], [], []
            for ci in sorted(cg.keys(), key=int):
                a = cg[ci].attrs
                row = dict(
                    pid=pid, pass_=pass_name, subset=subset, chunk=int(ci),
                    ns_original=int(a["ns_original"]),
                    cr_total=float(a["cr_total"]), cr_svd=float(a["cr_svd"]),
                    cr_wp=float(a["cr_wp"]), rmse=float(a["rmse"]),
                )
                per_chunk.append(row)
                cr_total.append(row["cr_total"])
                cr_svd.append(row["cr_svd"])
                cr_wp.append(row["cr_wp"])
                rmse.append(row["rmse"])

            per_pid.append(dict(
                pid=pid, pass_=pass_name, subset=subset,
                nc=nc, fs=fs, ns_total=ns_total, duration_s=ns_total / fs if fs else np.nan,
                epsilon=float(meta.get("epsilon", np.nan)),
                alpha=float(meta.get("alpha", np.nan)),
                n_chunks=len(cr_total),
                **counts, n_bad=n_bad,
                frac_bad=n_bad / nc if nc and not np.isnan(n_bad) else np.nan,
                **sat,
                cr_total_mean=float(np.mean(cr_total)) if cr_total else np.nan,
                cr_total_median=float(np.median(cr_total)) if cr_total else np.nan,
                cr_svd_mean=float(np.mean(cr_svd)) if cr_svd else np.nan,
                cr_wp_mean=float(np.mean(cr_wp)) if cr_wp else np.nan,
                rmse_mean=float(np.mean(rmse)) if rmse else np.nan,
                rmse_median=float(np.median(rmse)) if rmse else np.nan,
                rmse_max=float(np.max(rmse)) if rmse else np.nan,
            ))
    return per_pid, per_chunk


def make_figures(df_pid, df_chunk, fig_dir):
    """Write the QC summary figures to *fig_dir* with the dated prefix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    import addcopyfighandler  # noqa: F401

    sns.set_theme(context="notebook", style="whitegrid")
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        path = fig_dir.joinpath(f"{FIG_DATE}_lfp_qc_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  figure → {path}")

    # Saturation and bad channels are pass-independent → one row per PID.
    pid1 = df_pid.drop_duplicates("pid")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(pid1["saturated_fraction"] * 100, bins=40, ax=ax[0])
    ax[0].set(xlabel="saturated fraction [%]", ylabel="PIDs", title="ADC saturation per PID")
    ax[0].set_yscale("log")
    sns.histplot(pid1["total_saturated_sec"], bins=40, ax=ax[1])
    ax[1].set(xlabel="total saturated [s]", ylabel="PIDs", title="Saturated duration per PID")
    ax[1].set_yscale("log")
    save(fig, "saturation")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(pid1["frac_bad"] * 100, bins=40, ax=ax[0])
    ax[0].set(xlabel="bad channels [%]", ylabel="PIDs", title="Bad-channel fraction per PID")
    melt = pid1.melt(value_vars=["n_dead", "n_noisy", "n_outside"],
                     var_name="label", value_name="count")
    sns.boxplot(data=melt, x="label", y="count", ax=ax[1])
    ax[1].set(xlabel="", ylabel="channels per PID", title="Bad-channel counts by type")
    save(fig, "bad_channels")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df_pid, x="cr_total_mean", hue="pass_", bins=40, ax=ax[0], element="step")
    ax[0].set(xlabel="mean compression ratio", ylabel="PIDs", title="Compression ratio per PID")
    sns.histplot(df_pid, x="rmse_median", hue="pass_", bins=40, ax=ax[1], element="step")
    ax[1].axvline(25, color="k", ls="--", lw=1, label="25 µV target")
    ax[1].set(xlabel="median RMSE [µV]", ylabel="PIDs", title="Reconstruction RMSE per PID")
    save(fig, "compression")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(df_pid, x="cr_total_mean", y="rmse_median", hue="pass_", s=12, alpha=0.5, ax=ax)
    ax.axhline(25, color="k", ls="--", lw=1)
    ax.set(xlabel="mean compression ratio", ylabel="median RMSE [µV]",
           title="Compression ratio vs RMSE")
    save(fig, "cr_vs_rmse")


def main():
    parser = argparse.ArgumentParser(description="LFP compression QC report.")
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT,
                        help="Root holding the merged lf_compressed*all*.h5 archives.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the parquet tables (default: --local-root).")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR, help="Figure output directory.")
    parser.add_argument("--no-figures", action="store_true", help="Write tables only.")
    args = parser.parse_args()
    out_dir = args.out_dir or args.local_root

    archives = sorted(args.local_root.rglob("lf_compressed*all*.h5"))
    if not archives:
        raise SystemExit(f"No lf_compressed*all*.h5 under {args.local_root}")
    print(f"Scanning {len(archives)} archive(s):")
    for a in archives:
        print(f"  {a}")

    per_pid, per_chunk = [], []
    for a in archives:
        p, c = scan_archive(a)
        per_pid.extend(p)
        per_chunk.extend(c)

    df_pid = pd.DataFrame(per_pid).rename(columns={"pass_": "pass"})
    df_chunk = pd.DataFrame(per_chunk).rename(columns={"pass_": "pass"})

    out_dir.mkdir(parents=True, exist_ok=True)
    f_pid = out_dir.joinpath("lfp_qc_per_pid.pqt")
    f_chunk = out_dir.joinpath("lfp_qc_per_chunk.pqt")
    df_pid.to_parquet(f_pid)
    df_chunk.to_parquet(f_chunk)
    print(f"\nWrote {len(df_pid)} PID rows → {f_pid}")
    print(f"Wrote {len(df_chunk)} chunk rows → {f_chunk}")

    n_no_labels = int(df_pid["n_bad"].isna().sum())
    if n_no_labels:
        print(f"WARNING: {n_no_labels}/{len(df_pid)} PID-rows have no bad-channel labels "
              "(detection not fed through at compression)")

    # quick console summary per pass
    for pass_name, g in df_pid.groupby("pass"):
        g1 = g.drop_duplicates("pid")
        print(f"\n[{pass_name}]  {len(g)} recordings")
        print(f"  saturated fraction : median {g1['saturated_fraction'].median():.4%}  "
              f"max {g1['saturated_fraction'].max():.4%}")
        print(f"  bad channels       : median {g1['frac_bad'].median():.2%}  "
              f"max {g1['frac_bad'].max():.2%}")
        print(f"  compression ratio  : median {g['cr_total_median'].median():.0f}")
        print(f"  RMSE [µV]          : median {g['rmse_median'].median():.2f}  "
              f"max {g['rmse_max'].max():.2f}")

    if not args.no_figures:
        make_figures(df_pid, df_chunk, args.fig_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()