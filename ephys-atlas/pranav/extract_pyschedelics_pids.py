
from one.api import ONE
import pandas as pd
from pathlib import Path

one = ONE()
# Rest query for getting psychedelics insertions
insertions = one.alyx.rest('insertions', 'list', django='session__projects__name__icontains,psychedelics')

# Get pids from insertions
pids, alyx_pids = [item["id"] for item in insertions], insertions

# Get the corresponding eids, and probe_names
df= pd.DataFrame([{'pid':val['id'],'eid':val['session'],'probe_name':val['name']} for val in alyx_pids],columns=['pid','eid','probe_name'])

#TODO - Add exclude pids

#Output to a CSV file.
OUTPUT_DIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics")
df.to_csv(OUTPUT_DIR / 'psychedlics_pids.csv',index=False)

