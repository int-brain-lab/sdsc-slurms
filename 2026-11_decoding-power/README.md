# BWM decoding grid for the neuron-pool power analysis (Popeye)

Runs `projects/repeated-site-decoding-power/exploratory-analyses/005_decoding_grid.py` from an
`ibl-ai-agent` clone on Popeye. The science code, decoder and QC tables live in that project; this
folder only holds the submission script.

Workload: stimulus-side BWM decoding (L1 logistic, nested 5x5 CV, 1 run). 278 sessions, CTX-VIS/HPF/TH
plus 56 Beryl regions, 12,615 (session, region, k) cells, 20 true + 200 pseudo-session decodes each,
about 2.8 M decodes in total. About 150 core-hours on an M-series laptop core; expect roughly 4–6 h on one
48-core node, or 1–2 h with `--array=0-3`.

## Setup (once, on a Popeye login node)
```bash
cd ~/Documents && git clone git@github.com:int-brain-lab/ibl-ai-agent.git && cd ibl-ai-agent
module load python/3.11
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
# uv.lock is gitignored in ibl-ai-agent: copy the laptop's lock first (a fresh resolve pulls
# ibl-neuropixel 1.15, whose ibldsp no longer has sync_timestamps -> ibllib 4.0.1 import error)
#   (laptop) rsync -a ~/Documents/ibl-ai-agent/uv.lock popeye:Documents/ibl-ai-agent/
unset VIRTUAL_ENV && uv sync --frozen
uv run python scripts/download_datasets.py   # public BWM archives (~9 GB) -> reports/datasets/, writes data_locations.local.yaml
```
Then, **from the laptop**, copy the project folder. It is gitignored, so it is not part of the clone; it holds
the scripts plus `artifacts/003_spike_alignment_qc.csv`, which excludes misaligned insertions:
```bash
cd ~/Documents/ibl-ai-agent
rsync -av --exclude 'artifacts/005_*' --exclude '*.png' --exclude '*.parquet' \
  projects/repeated-site-decoding-power popeye:Documents/ibl-ai-agent/projects/
```
Check that the cluster copy uses the same dataset version as the laptop (`bwm_ephys` 1.2.1):
`ls ~/Documents/ibl-ai-agent/reports/datasets/bwm_ephys`.

## Run
```bash
cd ~/Documents/sdsc-slurms/2026-11_decoding-power
sbatch decoding_grid.sbatch                # one 48-core node
sbatch --array=0-3 decoding_grid.sbatch    # or up to 4 nodes (gen partition limit)
```
Stage 1 caches per-session inputs (`artifacts/005_inputs/<eid>.npz`, a few minutes); stage 2 writes
`artifacts/005_grid/<level>/<region>/<eid>.parquet`, coarse groups first. Resubmitting resumes.

## Progress / results
```bash
squeue -u $USER
find ~/Documents/ibl-ai-agent/projects/repeated-site-decoding-power/artifacts/005_grid -name '*.parquet' | wc -l   # 1,651 when done
```
Bring results back to the laptop:
```bash
rsync -av popeye:Documents/ibl-ai-agent/projects/repeated-site-decoding-power/artifacts/005_grid \
  ~/Documents/ibl-ai-agent/projects/repeated-site-decoding-power/artifacts/
```
