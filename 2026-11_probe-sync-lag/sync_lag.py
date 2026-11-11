"""Re-extract the spike times of the probe insertions whose imec sync edges are missing at the file start.

The 67 pids in `pids.csv` are the census recordings labelled `missing edges at start`: probe and nidq durations differ
by k half-periods of the 1 Hz imec sync, the edge counts differ by the same k, the edges are regular and complete, and
SpikeGLX < 20221012. ibllib `sync_probes.version3B` pairs the edges from the first one, which shifts the spikes by
k * 0.5 s. The correct pairing is from the last edge.

Steps (`--step`), each idempotent and per pid, a failure writes `<pid>/<step>.error` instead of raising:

- `extract`: imec sync QC (`ibldsp.sync.spikeglx_sync_qc`), corrected probe sync from the last-edge pairing, and
  re-interpolated `spikes.times` for every spike sorting collection holding `spikes.samples`. Writes the ALF tree
  `<pid>/raw_ephys_data/<probe>/*.sync.npy|*.timestamps.npy` and `<pid>/alf/<pname>/<sorter>/spikes.times.npy`,
  plus `<pid>/extract.json`.
- `wheel`: cross-correlation of the population rate with the wheel speed (wheel is on the nidq clock), on the original
  and on the re-extracted spike times. Writes `<pid>/wheel.json`.
- `aggregate`: `extract.pqt` and `wheel.pqt` in `OUTPUT_PATH`.

The BWM decoding of the 17 BWM pids is a separate SLURM array, see `decoding.py`.
"""

import os

os.environ["TQDM_DISABLE"] = "1"
import argparse
import json
import re
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from one.alf.exceptions import ALFObjectNotFound
from scipy.interpolate import interp1d
from scipy.signal import correlate

import ibldsp.sync
import spikeglx
from deploy.iblsdsc import OneSdsc as ONE
from ibllib.ephys.sync_probes import sync_probe_front_times

OUTPUT_PATH = Path("/mnt/home/owinter/ceph/sync_lag")
PIDS = pd.read_csv(Path(__file__).parent.joinpath("pids.csv"))
DEFAULT_SYNC_CHANNEL = {"nidq": 3, "probe": 6}  # ibllib 3B defaults, ephys_fpga.CHMAPS
H = 0.5  # half-period of the imec sync (s)
# wheel cross-correlation: bin (s), boxcar smoothing of the wheel speed (s), control baseline lag (s): median peak lag
# of 29 aligned control insertions, neural activity leads movement (spike-clock-offset-investigation 007)
DT, SMOOTH, WHEEL_BASELINE = 0.01, 0.05, -0.08

one = ONE(mode="local")


def load_stream(eid, collection, meta_pattern, stream):
    """
    Load the imec sync edges and the meta-data of one SpikeGLX stream.

    Parameters
    ----------
    eid : str
        Session UUID.
    collection : str
        ALF collection, `raw_ephys_data` or `raw_ephys_data/probeXX`.
    meta_pattern : str
        Meta file name pattern, `*.nidq.meta` or `*.ap.meta`.
    stream : str
        `nidq` or `probe`, selects the default sync channel.

    Returns
    -------
    numpy.ndarray
        Times (s) of the imec sync edges, from the start of the file.
    dict
        Meta-data.
    str
        Meta file dataset name (without the Flatiron UUID).
    """
    sync = one.load_object(eid, "sync", collection=collection, namespace="spikeglx")
    sync = {k.split(".")[0]: v for k, v in sync.items()}  # probe files carry the probe name: times.probe00
    meta_name = next(Path(d).name for d in one.list_datasets(eid, collection=collection) if Path(d).match(meta_pattern))
    md = spikeglx.read_meta_data(one.load_dataset(eid, meta_name, collection=collection, download_only=True))
    try:
        wiring = one.load_dataset(eid, "*.wiring.json", collection=collection)
        channel = spikeglx._sync_map_from_hardware_config(wiring).get("imec_sync", DEFAULT_SYNC_CHANNEL[stream])
    except Exception:  # noqa: BLE001, no wiring file: ibllib default
        channel = DEFAULT_SYNC_CHANNEL[stream]
    return sync["times"][sync["channels"] == channel], md, meta_name


def spike_collections(eid, pname):
    """Spike sorting collections of an insertion that hold `spikes.samples`, revision folders stripped (the default
    revision is loaded)."""
    dsets = one.list_datasets(eid, collection=f"alf/{pname}*")
    return sorted({re.sub(r"/#[^/]+#$", "", str(Path(d).parent)) for d in dsets if Path(d).name == "spikes.samples.npy"})


def apply_sync(sync_points, t):
    """Probe time to session time, as `ibllib.ephys.sync_probes.apply_sync`: linear interpolation, extrapolated."""
    return interp1d(sync_points[:, 0], sync_points[:, 1], fill_value="extrapolate")(t)


def original_spike_times(pid, eid, collection, rec, samples, sr):
    """
    Spike times as released, or as the pipeline would release them (first-edge pairing of ibllib version3B).

    In order: `spikes.times` of the collection; the samples through the released probe `*.sync.npy`; the samples
    through the first-edge pairing computed by `extract` (`<pid>/first_edge_pairing.sync.npy`), for probes never synced.

    Returns
    -------
    numpy.ndarray
        Spike times (s).
    str
        Source: `spikes.times`, `sync.npy` or `first_edge_pairing`.
    """
    try:
        return one.load_dataset(eid, "spikes.times.npy", collection=collection), "spikes.times"
    except ALFObjectNotFound:
        pass
    try:
        file_sync = one.load_dataset(eid, "*.sync.npy", collection=f"raw_ephys_data/{rec}", download_only=True)
        return apply_sync(np.load(file_sync), samples / sr), "sync.npy"
    except ALFObjectNotFound:
        sync_points = np.load(OUTPUT_PATH.joinpath(pid, "first_edge_pairing.sync.npy"))
        return apply_sync(sync_points, samples / sr), "first_edge_pairing"


def extract(pid, eid, pname):
    """
    Corrected probe sync and spike times from the last-edge pairing.

    Parameters
    ----------
    pid, eid, pname : str
        Probe insertion, session and probe name.

    Returns
    -------
    dict
        Sync QC, pairing and, per spike sorting collection, the median shift new - original spike times (s).
    """
    out_dir = OUTPUT_PATH.joinpath(pid)
    rec = re.sub(r"[a-d]$", "", pname)
    t_nidq, md_nidq, _ = load_stream(eid, "raw_ephys_data", "*.nidq.meta", "nidq")
    t_probe, md_probe, meta_name = load_stream(eid, f"raw_ephys_data/{rec}", "*.ap.meta", "probe")
    qc = ibldsp.sync.spikeglx_sync_qc(md_nidq, md_probe, t_nidq, t_probe, label=f"{pid} {pname}")
    # missing edges at start: only the duration and edge count criteria fail, by the same number of half-periods
    if not (qc["sync_present"] and qc["regular"] and qc["complete"] and qc["dn_edges"] == qc["k"] != 0):
        raise ValueError(f"not a missing-edges-at-start case: {qc}")
    n = min(t_nidq.size, t_probe.size)
    sr = spikeglx._get_fs_from_meta(md_probe)
    sync_points, qc_fit = sync_probe_front_times(t_probe[-n:], t_nidq[-n:], sr, display=False, type="smooth", tol=2.5)
    # the faulty pairing of ibllib version3B, reference for the probes that were never synced
    sync_first, qc_fit_first = sync_probe_front_times(t_probe[:n], t_nidq[:n], sr, display=False, type="smooth", tol=2.5)
    np.save(out_dir.joinpath("first_edge_pairing.sync.npy"), sync_first)
    # same file names and content as ibllib sync_probes._save_timestamps_npy
    raw_dir = out_dir.joinpath("raw_ephys_data", rec)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_sync = raw_dir.joinpath(meta_name.replace(".ap.meta", ".sync.npy"))
    np.save(file_sync, sync_points)
    np.save(raw_dir.joinpath(meta_name.replace(".ap.meta", ".timestamps.npy")), sync_points * np.array([sr, 1.0]))
    res = {"pid": pid, "eid": eid, "pname": pname, "fs": sr, "qc_fit": bool(qc_fit), "qc_fit_first_edge": bool(qc_fit_first), "n_pairs": n} | {
        k: (v.item() if isinstance(v, np.generic) else v) for k, v in qc.items()
    }
    for collection in spike_collections(eid, pname):
        samples = one.load_dataset(eid, "spikes.samples.npy", collection=collection)
        new = apply_sync(sync_points, samples / sr)
        alf_dir = out_dir.joinpath(collection)
        alf_dir.mkdir(parents=True, exist_ok=True)
        np.save(alf_dir.joinpath("spikes.times.npy"), new)
        times, source = original_spike_times(pid, eid, collection, rec, samples, sr)
        res[f"shift_s.{collection}"] = float(np.median(new - times))
        res[f"original_source.{collection}"] = source
    return res


def z(x):
    """Z-score."""
    return (x - x.mean()) / x.std()


def wheel_xcorr(spike_times, wheel, max_lag):
    """
    Peak lag of the cross-correlation of the population rate with the wheel speed, with a block bootstrap CI.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times (s) of all units, sorted.
    wheel : dict
        ALF wheel object (`timestamps`, `position`), nidq clock.
    max_lag : float
        Half-width of the lag window (s).

    Returns
    -------
    dict
        `peak_lag_s`, `offset_s` (peak lag minus the control baseline), `ci_lo`, `ci_hi`, `peak_r`, `peak_snr`.
    """
    edges = np.arange(max(wheel.timestamps[0], spike_times[0]), min(wheel.timestamps[-1], spike_times[-1]), DT)
    rate = np.histogram(spike_times, edges)[0].astype(float)
    pos = np.interp(edges, wheel.timestamps, wheel.position)
    speed = np.abs(np.convolve(np.diff(pos), np.ones(int(SMOOTH / DT)) / (SMOOTH / DT), mode="same"))
    nlag = int(max_lag / DT)
    lags = np.arange(-nlag, nlag + 1) * DT

    def xc(idx):
        full = correlate(z(rate[idx]), z(speed[idx]), mode="full", method="fft") / idx.size
        return full[idx.size - 1 - nlag : idx.size + nlag]

    xcf = xc(np.arange(rate.size))
    xcb = np.array([xc(b) for b in np.array_split(np.arange(rate.size), 10)])
    rng = np.random.default_rng(0)
    boot = [lags[np.argmax(xcb[rng.integers(0, 10, 10)].mean(0))] for _ in range(500)]
    iqr = np.percentile(xcf, 75) - np.percentile(xcf, 25)
    peak_lag = float(lags[np.argmax(xcf)])
    return {
        "peak_lag_s": peak_lag,
        "offset_s": peak_lag - WHEEL_BASELINE,
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "peak_r": float(xcf.max()),
        "peak_snr": float((xcf.max() - np.median(xcf)) / (iqr + 1e-12)),
    }


def wheel(pid, eid, pname):
    """Wheel cross-correlation offsets of the original and re-extracted spike times, first spike sorting collection
    in order of preference pykilosort, iblsorter, other."""
    collections = spike_collections(eid, pname)
    collection = sorted(collections, key=lambda c: ("pykilosort" not in c, "iblsorter" not in c, c))[0]
    k = PIDS.set_index("pid").loc[pid, "k"]
    max_lag = abs(k) * H + 2
    w = one.load_object(eid, "wheel", collection="alf")
    sr = json.loads(OUTPUT_PATH.joinpath(pid, "extract.json").read_text())["fs"]
    samples = one.load_dataset(eid, "spikes.samples.npy", collection=collection)
    original = np.sort(original_spike_times(pid, eid, collection, re.sub(r"[a-d]$", "", pname), samples, sr)[0])
    shifted = np.sort(np.load(OUTPUT_PATH.joinpath(pid, collection, "spikes.times.npy")))
    res = {"pid": pid, "collection": collection, "k": int(k), "expected_offset_s": float(k * H)}
    for label, st in {"original": original, "shifted": shifted}.items():
        res |= {f"{label}.{key}": v for key, v in wheel_xcorr(st, w, max_lag).items()}
    return res


def run_pid(rec, step, overwrite=False):
    """Run one step for one pid, write `<pid>/<step>.json` or `<pid>/<step>.error`."""
    out_dir = OUTPUT_PATH.joinpath(rec.pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_out, file_error = out_dir.joinpath(f"{step}.json"), out_dir.joinpath(f"{step}.error")
    if file_out.exists() and not overwrite:
        return
    file_error.unlink(missing_ok=True)
    try:
        # eid and probe name from pids.csv: pid2eid needs Alyx or an insertions cache table, absent on popeye
        if step == "extract":
            res = extract(rec.pid, rec.eid, rec.pname)
        elif step == "wheel":
            res = wheel(rec.pid, rec.eid, rec.pname)
        file_out.write_text(json.dumps(res, indent=1))
        print(rec.pid, step, "done", flush=True)
    except Exception:  # noqa: BLE001
        file_error.write_text(traceback.format_exc())
        print(rec.pid, step, "ERROR", flush=True)


def aggregate():
    """Concatenate the per-pid json of each step into `<step>.pqt` in OUTPUT_PATH, and list the errors."""
    for step in ["extract", "wheel"]:
        rows = [json.loads(f.read_text()) for f in sorted(OUTPUT_PATH.glob(f"*/{step}.json"))]
        if rows:
            df = PIDS.merge(pd.DataFrame(rows), on="pid", how="right")
            if step == "wheel":  # confident wheel estimate (007 criteria) that the re-extracted spikes are aligned
                df["wheel_confident"] = (df["shifted.peak_snr"] >= 2) & (df["shifted.ci_hi"] - df["shifted.ci_lo"] < 0.3)
                df["aligned"] = df["shifted.offset_s"].abs() < 0.15
            df.to_parquet(OUTPUT_PATH.joinpath(f"{step}.pqt"))
            print(f"{step}: {len(df)} rows")
        errors = sorted(OUTPUT_PATH.glob(f"*/{step}.error"))
        print(f"{step}: {len(errors)} errors", *[e.parent.name for e in errors], sep="\n  ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", choices=["extract", "wheel", "aggregate"], default="extract")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pids", nargs="*", help="subset of pids, default all of pids.csv")
    parser.add_argument("--bwm", action="store_true", help="only the BWM pids")
    args = parser.parse_args()
    if args.step == "aggregate":
        aggregate()
    else:
        todo = PIDS if not args.pids else PIDS[PIDS.pid.isin(args.pids)]
        todo = todo[todo.bwm] if args.bwm else todo
        joblib.Parallel(n_jobs=args.workers)(
            joblib.delayed(run_pid)(rec, args.step, args.overwrite) for rec in todo.itertuples()
        )
