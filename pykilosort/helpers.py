from pathlib import Path
import pandas as pd
from deploy.iblsdsc import OneSdsc as ONE


def check_pids(pids, one=None, user="clangfield"):
    if not one:
        one = ONE()
    quar_path = Path(f"/mnt/sdceph/users/ibl/data/quarantine/tasks_{user}/SpikeSorting/")
    status = []
    path = []
    time = []
    for pid in pids:
        eid, probe = one.pid2eid(pid)
        rel_path = one.eid2path(eid).relative_to(one.cache_dir)
        session_path = quar_path.joinpath(rel_path)
        probe_path = session_path.joinpath("raw_ephys_data", probe)
        
        alf = session_path.joinpath("alf", probe, "pykilosort")
        if alf.joinpath("waveforms.channels.npz").exists():
            status.append("Done")
        elif probe_path.exists():
            status.append("Started")
        else:
            status.append("Not started")
        path.append(probe_path)
        
        logfile = alf.joinpath("_ibl_log.info_pykilosort.log")
        if logfile.exists():
            with open(logfile, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                time_s = last_line.split()[-1].split("s")[0]
                time_m = float(time_s) / 60
                time_h = round(time_m / 60, 2)
                time.append(time_h)
        else:
            time.append(np.nan)
        
    df = pd.DataFrame({"path":path, "status":status, "duration_hours":time},index=pids)
    return df

def check_pids_registered(pids, one=None):
    if not one:
        one = ONE()
    status = []
    
    for pid in pids:
        eid, probe = one.pid2eid(pid)
        if f'alf/{probe}/pykilosort/#2024-05-06#/spikes.samples.npy' in one.list_datasets(eid):
            status.append("Yes")
        else:
            status.append("No")
            
    df = pd.DataFrame({"registered":status}, index=pids)
    
    return df
    