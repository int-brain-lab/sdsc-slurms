# LFP-compression / saturation region-decoding experiment (SDSC)

Extracts LF/CSD features over the **ephys-atlas cohort** (every
histology-resolved insertion compressed by `../2026-06-lfpack/compress.py` —
a strict superset of the curated ~700-PID BWM behavioural cohort used by
`../2026-07_lfp-encoders`), trains the canonical XGBoost region classifier
(Cosmos-level labels), and compares held-out accuracy:

1. **Compression impact**: across the three LFP sources below.
2. **Saturation impact**: across two snippet-selection modes (naive vs
   saturation-avoided), per source.

| `--lfp-source` | LFP read from | notes |
|----------------|---------------|-------|
| `default`      | `lf_compressed_all.h5` (ephys-atlas superset, private S3) | lfpack SVD+WP ε=150, α=28 |
| `aggressive`   | `lf_compressed_aggressive_all.h5` (ephys-atlas superset, private S3) | lfpack SVD+WP ε=450, α=96 |
| `uncompressed` | `<pid>/lf_resampled_car_cadzow.npy` (Cadzow checkpoint, pre-SVD/WP) | already on the cluster, no download needed |

**Not** the public `resources/ibl-agent-data/bwm/lf_compressed_all_bwm.h5`
flagship archive used by `2026-07_lfp-encoders` — that only covers the
curated BWM subset, not the full ephys-atlas cohort this project targets.

**Not** `2026-06-lfpack/compress.py`'s own per-PID output tree
(`/mnt/home/owinter/ceph/ea/denoised_lfp/lf_compressed_all.h5` etc.) either —
verified 2026-08-31 that copy is **not region-annotated** (`channels` there
has only `lateral_um`/`axial_um`/`labels`, no `x`/`y`/`z`/`atlas_id`/`acronym`)
because `attach_ibl_metadata.py` only ever ran against a locally-rsynced copy,
never published back to that tree. The properly annotated archive lives on S3
(`ephysatlas.data.download_lfp_features`, confirmed byte-identical to the
local annotated copy at `~/Documents/datadisk/lfp-processing/lfpack/v03/`) and
is what `extract.py --download` fetches — see Setup below.

PID list and snippet grid are derived directly from the archive
(`LFPackReader.recordings`, and each PID's own duration) rather than the
legacy `../ephys-atlas/snippets_*.csv` files. The snippet grid itself (600 s
spacing, 200 s start, ~8.79 s window) matches that legacy
`ephys_atlas.features` convention — "the default eatools" snippet
configuration.

## Files
- `extract.py` — driver: source/mode-parameterized, PID list from the
  archive, joblib over PIDs. `LFP_DATA_ROOT` / `CADZOW_ROOT` / `OUTPUT_ROOT`
  at the top — confirm these paths on the cluster before running.
- `extract.sbatch` — single-node job (feature extraction is far cheaper per
  PID than a regression fit, so no array striping like `2026-07_lfp-encoders`).
- `saturation_snippets.py` — builds `snippets_avoiding_saturation.csv` for
  `--snippet-mode saturation-avoided`.
- `aggregate.py` — thin wrapper around `ephysatlas.aggregation.produce_output_dataframes`.
- `train_classifier.py` — 5-fold PID-grouped `XGBClassifier`, Cosmos labels,
  LF-only feature set (`voltage_features_set(["raw_lf","raw_lf_csd"])`),
  mirroring `packages/ibleatools/examples/training_region_predictor_gradient_boosting.py`.
- `report.py` — run **locally**, produces the two comparison barplots.

## Setup (before first submit)
1. **Download the region-annotated compressed archives once** (~35 GB total;
   fast over the cluster's own AWS bandwidth — takes minutes, not hours):
   ```bash
   python extract.py --download
   ```
   This calls `ephysatlas.data.download_lfp_features(LFP_DOWNLOAD_ROOT,
   project="ibl_neuropixel_brainwide_01", level=...)` for both tiers, landing
   at `LFP_DATA_ROOT` = `LFP_DOWNLOAD_ROOT/ibl_neuropixel_brainwide_01/lfp_aggregates/`
   (`/mnt/home/owinter/ceph/lfp-decoders/...` by default — confirm at the top
   of `extract.py`). Needs Alyx-issued AWS credentials (private bucket) via an
   authenticated `OneSdsc`/`ONE` instance — already the case on Popeye.
   `CADZOW_ROOT` (the per-PID Cadzow checkpoints) needs no download: it's
   already on the cluster at `2026-06-lfpack/compress.py`'s own output tree.
2. For the saturation experiment, build the saturation-avoiding snippet grid
   once (cheap — only reads each PID's small saturation table):
   ```bash
   python saturation_snippets.py
   ```

## Run
Always via `sbatch` (a compute node), same reasoning as `2026-07_lfp-encoders`
— avoid OOM/CPU contention on a login node.

```bash
# fast smoke: few PIDs, validates the whole path (extract -> aggregate -> train)
sbatch extract.sbatch --lfp-source default --snippet-mode all --limit 4

# full run: once per (source, mode) combination — 6 submissions total
for source in default aggressive uncompressed; do
  sbatch extract.sbatch --lfp-source "$source" --snippet-mode all
  sbatch extract.sbatch --lfp-source "$source" --snippet-mode saturation-avoided
done
```
Resumable: a snippet whose feature files already exist is skipped unless
`--overwrite` is passed.

Then, once each `extract.sbatch` job finishes, aggregate and train (cheap
enough to run directly on a login node, or wrap in its own small sbatch job if
memory-constrained):
```bash
for source in default aggressive uncompressed; do
  for mode in all saturation-avoided; do
    python aggregate.py --lfp-source "$source" --snippet-mode "$mode"
    python train_classifier.py --lfp-source "$source" --snippet-mode "$mode"
  done
done
```

## Check progress
```bash
squeue -u $USER   # empty = nothing still queued/running
for source in default aggressive uncompressed; do
  for mode in all saturation-avoided; do
    d="$OUTPUT_ROOT/$source/$mode"
    n=$(ls "$d"/*/channels.pqt 2>/dev/null | wc -l)
    echo "$source/$mode: $n PIDs done"
  done
done
```

## Archive for transfer
```bash
cd "$OUTPUT_ROOT"
tar czf lfp-decoders-results.tar.gz */*/accuracy.json */*/predictions.pqt 2>/dev/null
```
Then from the laptop:
```bash
cd /Users/olivier/Documents/datadisk/lfp-processing/lfp-decoders
rsync --progress -av -e ssh popeye:~/ceph/lfp-decoders/results/lfp-decoders-results.tar.gz ./
tar xzf lfp-decoders-results.tar.gz
python report.py --results-root .
```

## Model (locked)
Same feature-computation engine as the production `ephys_atlas.features`
pipeline (`ephysatlas.feature_calculators`/`feature_computation`), restricted
to the LF/CSD feature families (these sources have no AP stream). Since the
lfpack/Cadzow LF is already destriped/CAR'd/decimated/Cadzow-denoised
upstream, every snippet is computed with `skip_lf_destripe=True` and
`CsdParams(decimate=1, denoise=False)` — see
`ephysatlas.feature_calculators.lfpack`/`.numpy_array`'s module docstrings for
why (re-destriping an already-decimated signal outright crashes; re-decimating
an already-250 Hz signal by another factor of 10 also crashes in
`scipy.signal.decimate`).
