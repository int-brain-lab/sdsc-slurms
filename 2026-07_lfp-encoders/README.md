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
```bash
# smoke test: a few PIDs, one source
sbatch encode.sbatch --lfp-source default --limit 8 --workers 4
# full run per source (single node as configured; widen --array in the sbatch to stripe across nodes)
sbatch encode.sbatch --lfp-source default
sbatch encode.sbatch --lfp-source aggressive
sbatch encode.sbatch --lfp-source uncompressed
```
Resumable: PIDs with an existing `<pid>_band.parquet` under the source's outdir are skipped.

Each worker builds **one** `OneSdsc` and reuses it for every PID it handles (rather than
reconnecting per PID), and its first connection is jittered by up to `--stagger` seconds
(default 30) so the workers don't all hit alyx at job start. Lower it for a quick smoke
test; raise it if you fan out across more nodes.

## Outputs (under `OUTPUT_ROOT/<source>/`)
`basis.npz`, `model_config.json`, per-PID `scores/<pid>_<kind>.parquet` +
`kernels/<pid>_<kind>.npz`. Scores carry `has_wheel`/`has_pupil` flags and per-group
drop-R². The three source dirs are directly comparable (identical design). Pool with
`results_io.load_scores(OUTPUT_ROOT/<source>)`.

## Model (locked)
Events core (stimOn/move/feedback) always fit; **wheel** and **pupil** are gated add-ons;
**paw dropped**. Two target families (raw broadband, band-power envelopes: delta/theta/beta/gamma).
±1.5 s raised-cosine lags, per-group Tikhonov, 5-fold CV, 30-perm circular-shift null.