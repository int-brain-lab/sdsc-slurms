# Resume prompt: produce + publish the BWM version of the mild lfpack compression tier

Resume the LFP-compression mild-tier project (ephys-atlas cohort) — the BWM-variant leg.
Load memory topic "project_lfp-decoders" first for full context (design history, gotchas,
git policy, exact status of every step).

## Status so far (full ephys-atlas/non-BWM leg — DONE)

A 4th lfpack compression tier "mild" (epsilon=100, alpha=14) has been fully produced,
annotated, and (as of this prompt) is being published for the **full ephys-atlas cohort**
(1099 PIDs, project `ibl_neuropixel_brainwide_01`), not the curated BWM cohort:

1. `compress.py`/`reduce.py` code changes (PARAMS/PASSES + a resumable-sentinel bug fix) —
   committed+pushed to `sdsc-slurms` `origin/main` (`e04b248`, `cfbcd60`).
2. `compress.sbatch` computed `mild` for all 1099 PIDs (done 2026-09-02).
3. `reduce.py --passes mild` merged all 1099 PIDs into
   `/mnt/home/owinter/ceph/ea/denoised_lfp/lf_compressed_mild_all.h5` (32G) on Popeye (done
   2026-09-03).
4. Rsynced that file to `~/Documents/datadisk/lfp-processing/lfpack/v03/` on this Mac
   (md5-verified identical).
5. `attach_ibl_metadata.py --local-root .../lfpack/v03` (dry-run then real) — 0 failures
   across every bucket (channels/sync-missing/sync-invalid/geometry-mismatch/unexpected),
   all 1099 PIDs. **Scoping note**: deliberately pointed `--local-root` at `v03/` only, not
   the script's own broader default `LOCAL_ROOT` (`.../lfpack`), which also globs unrelated
   older archives under `latemute/`, `sample10/`, `sample10_copy/`, `v03_bwm/` — touching
   those would have been idempotent-safe but wasted ~80min of redundant Alyx/ONE network
   calls per pass.
6. Publishing `v03/lf_compressed_mild_all.h5` to the **private** S3 bucket
   `ibl-brain-wide-map-private`, key
   `aggregates/atlas/projects/ibl_neuropixel_brainwide_01/lfp_aggregates/lf_compressed_mild_all.h5`,
   via `aws s3 cp --profile ibl --region us-east-1 ...` — hit a real gotcha: **two upload
   attempts both failed** with `Connection was closed before we received a valid response`
   mid-multipart-upload (once at ~213MB/31.8GB, once near-complete at partNumber 863) — a
   flaky-network issue on this Mac's connection during a long sustained multipart transfer,
   not a credentials/bucket/key problem (bucket, key, and `ibl` profile all confirmed correct
   by listing the existing `default`/`aggressive` objects at the same prefix beforehand).
   Retrying with `aws configure set default.s3.max_concurrent_requests 4 --profile ibl` +
   `AWS_RETRY_MODE=adaptive AWS_MAX_ATTEMPTS=10` for resilience. **Verify this actually
   succeeded before assuming so** — check with:
   ```
   aws s3api head-object --profile ibl --region us-east-1 \
     --bucket ibl-brain-wide-map-private \
     --key aggregates/atlas/projects/ibl_neuropixel_brainwide_01/lfp_aggregates/lf_compressed_mild_all.h5
   ```
   (a 404 means it's still not there — don't trust a background task's reported exit code
   alone if the command was chained with a trailing `echo $?`, which masks the real exit
   status — this bit us once already this session).

## What this prompt is actually for: the BWM leg (NOT yet started)

Everything above is the **full ephys-atlas cohort** artifact. IBL normally *also* publishes a
second, separate artifact scoped to the curated **BWM cohort** (~700 PIDs) — this exists
today for `default`/`aggressive` (see `v03_bwm/lf_compressed_all_bwm.h5` and
`v03_bwm/lf_compressed_aggressive_all_bwm.h5` locally) but **does not yet exist for `mild`**.
None of the steps below have been run yet.

**Key difference from the ephys-atlas leg**: the BWM artifact publishes to a **different,
public** bucket/path than the ephys-atlas one — confirmed from
`sdsc-slurms/2026-07_lfp-encoders/encode.py`:
- bucket: `ibl-brain-wide-map-public` (not `-private`)
- key: `resources/ibl-agent-data/lf_compressed_mild_all_bwm.h5` (flat, no per-project path)

### Steps to execute, in order

1. **Merge the BWM subset on Popeye.** The per-PID `lf_compressed_mild.h5` files already
   exist for all 1099 PIDs (from the earlier full-cohort compress step) — this is a filtered
   merge only, no recomputation needed:
   ```
   ssh popeye  # confirm control socket alive first: ssh -O check popeye
   cd /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack
   nohup python3 reduce.py --passes mild --bwm > /tmp/reduce_mild_bwm.log 2>&1 < /dev/null & disown
   ```
   Output: `lf_compressed_mild_all_bwm.h5` under
   `/mnt/home/owinter/ceph/ea/denoised_lfp/`. `reduce.py --bwm` reports missing BWM PIDs (if
   any per-PID mild file is absent) as warnings — check the log for any before proceeding.
   Expect roughly proportional runtime to the full-cohort merge scaled by cohort size
   (~700/1099 of ~4h ≈ 2.5h), but this is a guess — watch actual progress
   (`tail -c 400 /tmp/reduce_mild_bwm.log | tr '\r' '\n' | tail -1`), don't just wait blindly.
   The full-cohort run's launcher-wrapper-hangs-past-120s-timeout-while-actually-fine quirk
   (see [[project_lfp-decoders]] SSH-drop gotcha) may recur — check `pgrep -af reduce.py` +
   the log before assuming a failed launch.

2. **Rsync the BWM archive down to this Mac**, into the existing `v03_bwm/` directory
   (matching the convention the default/aggressive BWM files already use):
   ```
   rsync -ah --progress popeye:/mnt/home/owinter/ceph/ea/denoised_lfp/lf_compressed_mild_all_bwm.h5 \
     /Users/olivier/Documents/datadisk/lfp-processing/lfpack/v03_bwm/
   ```
   Verify with md5 both ends before trusting the copy (`md5 -q` locally, `md5sum` on Popeye).
   Check local disk space first (`df -h /`) — the BWM file should be roughly 700/1099 of 32G
   ≈ 20G, smaller than the full-cohort one, but confirm actual size after step 1.

3. **Annotate it**, same tool, scoped to `v03_bwm/` only (not the broader default root, same
   reasoning as step 5 of the ephys-atlas leg — avoid touching unrelated archives):
   ```
   cd /Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-06-lfpack
   /Users/olivier/PycharmProjects/ephys-atlas/.venv/bin/python attach_ibl_metadata.py \
     --local-root /Users/olivier/Documents/datadisk/lfp-processing/lfpack/v03_bwm --dry-run
   ```
   Check the dry-run's failure-bucket summary (all should read 0, matching the ephys-atlas
   leg's clean result) before re-running without `--dry-run` to actually write. Runtime should
   scale down proportionally from the ephys-atlas leg's ~80min/pass (~700/1099 ≈ 50min/pass).

4. **Publish to the public bucket**, `ibl` profile, explicit region (same flakiness risk as
   the ephys-atlas leg — expect to possibly need a retry with reduced concurrency):
   ```
   aws s3 cp --profile ibl --region us-east-1 --only-show-errors \
     /Users/olivier/Documents/datadisk/lfp-processing/lfpack/v03_bwm/lf_compressed_mild_all_bwm.h5 \
     s3://ibl-brain-wide-map-public/resources/ibl-agent-data/lf_compressed_mild_all_bwm.h5
   ```
   Verify with `aws s3api head-object` (or `aws s3 ls` on the `resources/ibl-agent-data/`
   prefix) afterward — don't trust a bare exit code from a chained shell command; check the
   object directly.

5. Once both ephys-atlas and BWM `mild` archives are confirmed published, resume the
   original 11-step plan's remaining items (steps 7-11): add `'mild'` to
   `ephysatlas.data._LFP_AGGREGATES_FILES` (must land on `packages/ibleatools`'s
   `feature/stpc-refactor` branch, never `main`) and to `sdsc-slurms/2026-08_lfp-decoders/
   extract.py`'s `COMPRESSED_FILES`; re-run the lfp-decoders extract/aggregate/train pipeline
   for `mild` only; update `report.py`'s `SOURCE_ORDER`/`SOURCE_LABELS`. Full detail in
   [[project_lfp-decoders]].

## Operating notes carried over (all detailed in memory)

- `ssh -O check popeye` before any cluster command; if dead, ask the user to reconnect
  (`!ssh -fN popeye`) — never retry in a loop.
- Launch long remote commands detached on the cluster itself (`nohup ... & disown`), never as
  a plain foreground `ssh popeye "long command"`.
- Never chain `; echo EXIT_CODE=$?` (or similar) after a command run via `run_in_background`
  — it makes the wrapper's own reported exit code always 0, masking real failures. Check the
  actual command's output/exit status directly instead.
- Commit each logical feature as its own commit; `sdsc-slurms` commits go straight to
  `origin/main`, `packages/ibleatools` commits must go on `feature/stpc-refactor`.
