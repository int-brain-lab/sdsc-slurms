# Brain-wide LFP-encoding fit (SDSC)

Fits the lagged LFP←behaviour encoding model over all BWM insertions. Run **three
times, once per LFP source**, to measure how lossy compression affects the
recoverable behaviour signal (R² / drop-R²):

| `--lfp-source` | LFP read from | notes |
|----------------|---------------|-------|
| `default`      | `lf_compressed_all_bwm.h5` (consolidated, S3) | lfpack SVD+WP ε=150, α=28 |
| `aggressive`   | `lf_compressed_aggressive_all_bwm.h5` (consolidated, S3) | lfpack SVD+WP ε=450, α=96 |
| `uncompressed` | `cells/<pid>/lf_resampled_car_cadzow.npy` | Cadzow checkpoint (250 Hz CAR, pre-SVD/WP) — **always this, never raw .cbin** |

The two **compressed** tiers each read from a single consolidated multi-recording
HDF5 archive, keyed by PID (`recording=pid`) — the BWM lfpack files on the IBL public
S3 bucket under `resources/ibl-agent-data/`. The **uncompressed** reference stays
per-PID: the Cadzow checkpoint from `../2026-06-lfpack`, which borrows its time base +
channel metadata from the default archive so all three tiers share one 250 Hz grid and
channel layout.

All three share the **same design** (same behaviour, lags, CV, null); only the LFP
targets differ, so per-PID R² differences are attributable to compression.

Behaviour is loaded **from ONE** (`OneSdsc` local mirror) via `behavior_one.py`, not
the `bwm_behavior` shards — so wheel is complete for every session (avoids the shard
wheel gap, int-brain-lab/ibl-ai-agent#18).

## Files
- `encode.sbatch` — array job, one 48-core node per task, stripes PIDs `i::array_count`.
- `encode.py` — driver: source-parameterized, PID list from `cells/`, behaviour via ONE, joblib over PIDs.
- `behavior_one.py` — ONE-backed `load_trials_one` / `load_continuous_one` (wheel@100 Hz + pupil, gated).
- `design.py` · `targets.py` · `solve.py` · `results_io.py` · `lfpack_io.py` — shared science core (the
  source of truth; the laptop quarto repo imports these via `sys.path`). Co-located with `encode.py` so
  imports work from any CWD.

## Setup (before first submit)
1. Confirm the SDSC paths at the top of `encode.py` (`LFP_DATA_ROOT`, `LFP_CELLS_ROOT`,
   `OUTPUT_ROOT`).
2. **Download the compressed archive(s) once** (racy across array tasks, so do it up
   front on a login node — idempotent, skips if already complete):
   ```bash
   python encode.py --download --lfp-source default
   python encode.py --download --lfp-source aggressive
   python encode.py --download --lfp-source uncompressed
   ```
   They land flat under `LFP_DATA_ROOT/` (e.g. `lf_compressed_all_bwm.h5`) via
   `one.remote.aws.s3_download_file` from the IBL public bucket. The `uncompressed`
   source needs no download of its own but reuses the **default** archive for
   grid/channels, so `--download --lfp-source uncompressed` fetches `default`.
3. Confirm ONE on the cluster: `one.pid2eid(pid) -> (eid, label)`; pupil frame times via
   `SessionLoader.load_pose(views=['left'])` aligned to `load_pupil()`.
4. Uncompressed reader (`read_uncompressed`) reads `lf_resampled_car_cadzow.npy` and
   borrows tvec + channel metadata from the default archive. Verify on the cluster that
   the checkpoint's orientation and channel binning match the lfpack reader's `nc`
   targets (the code sums adjacent electrodes to `nc`; assert shapes on a smoke run).

## Run
Always via `sbatch` (a compute node): the band family builds a multi-GB `Y` per worker
and will get OOM-killed on a login node. Download the archive first (see Setup).

**Lambda selection defaults to `--lambda-mode per-band`** (one lambda fit per band
instead of one pooled lambda for all ~288 targets) — see "Lambda fitting" below before
resweeping; pass `--lambda-mode pooled` to reproduce the original `results_bwm_cluster`
behaviour exactly.
Set once per run, reused by every command below (Run/Check progress/Archive):
```bash
OUTDIR=~/ceph/lfp-encoders/results_bwm_v01_smart
```
```bash
# fast smoke: few PIDs, cheap null — validates the whole path (design→targets→solve→save)
sbatch encode.sbatch --lfp-source default --limit 4 --workers 4 --n-perm 2 --stagger 2
# full run per source (single node as configured; widen --array in the sbatch to stripe across nodes)
# --outdir: point at a NEW directory, don't overwrite an already-archived run
sbatch encode.sbatch --lfp-source default --outdir "$OUTDIR"
sbatch encode.sbatch --lfp-source aggressive --outdir "$OUTDIR"
sbatch encode.sbatch --lfp-source uncompressed --outdir "$OUTDIR"
```
Resumable: PIDs with **both** `<pid>_band.parquet` and `<pid>_raw.parquet` under the
source's outdir are skipped (a PID interrupted mid-fit re-runs).
(The fast smoke's `--n-perm 2` scores are throwaway — overwrite them with the full run, or
point `--outdir` elsewhere for the smoke.)

## Check progress
A single lfp-source encoding run should take around 6h30 to 6h45 on a full node and 221 Gb memory.

```bash
squeue -u $USER   # empty = nothing still queued/running
for s in default aggressive uncompressed; do
  b=$(ls "$OUTDIR/$s/scores/"*_band.parquet 2>/dev/null | wc -l)
  r=$(ls "$OUTDIR/$s/scores/"*_raw.parquet 2>/dev/null | wc -l)
  echo "$s: band=$b raw=$r"
done
```
Expected output: 
```shell
default: band=699 raw=699
aggressive: band=699 raw=699
uncompressed: band=699 raw=699
```

## Archive for transfer

```bash
cd "$(dirname "$OUTDIR")"
tar czf "$(basename "$OUTDIR").tar.gz" "$(basename "$OUTDIR")/"
```
Then from the laptop (use the archive name printed above):
```bash
cd /Users/olivier/Documents/datadisk/lfp-processing/lfp-encoders
rsync --progress -av -e ssh popeye:~/ceph/lfp-encoders/results_bwm_v01_smart.tar.gz ./
```

## Lambda fitting
`select_lambda` (pooled) picks one lambda per `(PID, kind)` by maximising the *median*
held-out R² across all ~288 targets at once — blind to a collapsing minority, and
occasionally lets a whole insertion catastrophically overfit under compression (see
`PLAN.md`/`index.qmd` "Result 5"). Fixed by:
- `select_lambda_robust` — same candidate sweep, tail-aware objective (mean of R²
  clipped to `[-1,1]`, not the raw median) plus a worst-case-quantile safety gate,
  falling back to the largest grid lambda if nothing clears it.
- `solve_encoding_grouped` / `permutation_null_r2_grouped` — actually *fit* a separate
  lambda per band (one Cholesky solve per group) instead of forcing one lambda across
  every band; exact-match-validated against the pooled functions when given a single
  group.
`encode.py --lambda-mode {per-band,pooled}` switches between them; `fit_pid`'s
`lambda_mode` param does the same for direct calls. Each score row's `lam` column
carries that row's own band's lambda now (see `results_io.save_pid_result`).

Each worker builds **one** `OneSdsc` and reuses it for every PID it handles (rather than
reconnecting per PID), and its first connection is jittered by up to `--stagger` seconds
(default 30) so the workers don't all hit alyx at job start. Lower it for a quick smoke
test; raise it if you fan out across more nodes.

## Saturation row exclusion
Saturated (ADC-clipped) LFP spans are excluded from the fit entirely (both training and
CV scoring) rather than trusted as muted-to-zero data -- fixes the `uncompressed`-tier
collapse regression seen in `v01` (a CV fold landing on a muted span has near-zero
target variance, so R² swings hugely negative; see `index.qmd` "Result 6"). The mask is
built once per PID from that source's own saturation table (`--saturation-margin-s`,
default `0.0` -- real-trace validation found no benefit from padding beyond the stored
interval, see `encode.SATURATION_MARGIN_S`) and applied identically to both target
kinds. A PID/source whose row exclusion leaves a CV fold too thin (`encode.
MIN_VALID_PER_FOLD`) errors out for that PID rather than fitting on a degenerate fold --
same per-PID try/except as any other fit error, so it doesn't stop the batch.

## Outputs (under `OUTPUT_ROOT/<source>/`)
`basis.npz`, `model_config.json`, per-PID `scores/<pid>_<kind>.parquet` +
`kernels/<pid>_<kind>.npz`. Scores carry `has_wheel`/`has_pupil` flags, `n_valid` (rows
actually used after saturation exclusion) and per-group drop-R². The three source dirs
are directly comparable (identical design). Pool with
`results_io.load_scores(OUTPUT_ROOT/<source>)`.

## Model (locked)
Events core (stimOn/move/feedback) always fit; **wheel** and **pupil** are gated add-ons;
**paw dropped**. Two target families (raw broadband, band-power envelopes: delta/theta/beta/gamma).
±1.5 s raised-cosine lags, per-group Tikhonov, 5-fold CV, 30-perm circular-shift null.
