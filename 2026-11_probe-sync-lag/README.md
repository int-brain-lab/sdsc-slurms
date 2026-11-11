# Probe sync lag: re-extraction of the spike times

67 probe insertions (`pids.csv`, 17 in the BWM release) whose imec sync edges are missing at the start of one file:
probe and nidq `fileTimeSecs` differ by k half-periods of the 1 Hz sync, the edge counts by the same k, the edges are
regular and complete, SpikeGLX < 20221012. ibllib `version3B` pairs the edges from the first one, which offsets the
spikes by k × 0.5 s. The fix is to pair from the last edge. `pids.csv` comes from
`oliche-quarto/analyses/2026-11-probe-sync-lag/artifacts/010/014_sync_issues.csv` (`issue == "missing edges at start"`;
2 more recordings of that label have no Alyx insertion: MFD_07 2023-08-30 probe00, KS019 2020-03-04 probe00).

## Paths
- code: `/mnt/home/owinter/Documents/sdsc-slurms/2026-11_probe-sync-lag`
- output: `/mnt/home/owinter/ceph/sync_lag`, one ALF-like folder per pid:
  - `raw_ephys_data/<probe>/*.sync.npy`, `*.timestamps.npy`: corrected probe sync, same format as ibllib
  - `alf/<pname>/<sorter>/spikes.times.npy`: re-extracted spike times, every collection holding `spikes.samples`
  - `first_edge_pairing.sync.npy`: the faulty ibllib pairing, reference for probes that were never synced
  - `extract.json`, `wheel.json` or `<step>.error` (traceback)
- `extract.pqt`, `wheel.pqt`, `decode.pqt`: aggregates, `decoding/<condition>/<target>/`: prior-localization pickles

## Environment
Needs ibllib, ibl-neuropixel on branch `sync-criteria` (`ibldsp.sync.spikeglx_sync_qc`), iblscripts (`deploy.iblsdsc`),
and for the decoding prior-localization on branch `sirena`, paper-brain-wide-map, behavior_models.
```bash
cd /mnt/home/owinter/Documents && module load python/3.11
uv venv sync-lag/.venv -p 3.11 && source sync-lag/.venv/bin/activate
uv pip install -e ibllib -e ibl-neuropixel -e iblscripts -e paper-brain-wide-map -e prior-localization \
  "behavior_models @ git+https://github.com/int-brain-lab/behavior_models.git" torch wfield psychofit scikit-learn
```

## Run
```bash
sbatch sync_lag.sbatch --step extract      # sync QC, last-edge pairing, spikes.times (minutes)
sbatch sync_lag.sbatch --step wheel        # wheel / population rate xcorr, original and re-extracted spikes
python sync_lag.py --step aggregate        # extract.pqt, wheel.pqt, lists the errors
python decoding.py --stage                 # login node: BWM trials table (saturation flags) from AWS
sbatch decoding.sbatch                     # 3 full nodes, 544 tasks (408 BWM + 136 pLeft), needs extract
python decoding.py --aggregate             # decode.pqt
```
Every step skips what is done (`--overwrite` to recompute), failed pids leave `<step>.error`.

## Checks
- `extract`: only the duration and edge-count criteria may fail, with `dn_edges == k`, otherwise the pid errors.
  `shift_s.<collection>`: median of new − original spike times, expected −k × 0.5 s.
- `wheel`: `original.offset_s` ≈ k × 0.5 s and `shifted.offset_s` ≈ 0 (baseline −0.08 s removed). `aligned` and
  `wheel_confident` flag the verdict.
- `decode`: BWM settings (`decoding.py` docstring). `original` reproduces the release; `shifted` changes only the spike
  times of the affected probe. Per session × region: score (mean balanced accuracy over 10 runs), median of the 200
  pseudo-session scores, p-value.

Tested locally (remote ONE in place of `OneSdsc`) on CSP004 2019-11-26 probe01 and NYU-12 2020-01-21 probe01 (k = −2):
wheel offset −0.97 → +0.03 s and −1.01 → −0.01 s; decoding run end to end with 1 pseudo-session.
