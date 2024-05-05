from pathlib import Path
from ibllib.pipes.ephys_tasks import SpikeSorting
from deploy.iblsdsc import OneSdsc as ONE
import sys

pid = sys.argv[1]
one = ONE()

SDSC_ROOT = Path("/mnt/sdceph/users/ibl/data/")
eid, pname = one.pid2eid(pid)
rel_path = one.eid2path(eid).relative_to(one.cache_dir)
session_path = SDSC_ROOT.joinpath(rel_path)

if __name__ == "__main__":
    print(f"1.7.0 run on {pid}")
    print(session_path)
    ssjob = SpikeSorting(session_path, one=None, pname=pname, device_collection='raw_ephys_data', location="popeye")
    ssjob.run()

    assert ssjob.status == 0