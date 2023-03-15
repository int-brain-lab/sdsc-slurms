from pathlib import Path
from one.api import ONE
one = ONE()

ROOT_PATH_POPEYE = Path("/mnt/sdceph/users/ibl/data")
DIR_OUT = Path("/mnt/home/owinter/ceph/microphone")
JOB_PREFIX = f"#DISBATCH PREFIX /usr/bin/time -f '%M %E' python check_microphone.py {DIR_OUT}"


fr_query = (
    'data_repository__name__istartswith,flatiron,'
    'exists,True,'
    'relative_path__iendswith,.flac,'
    'dataset__session__task_protocol__icontains,ephys'
)
frs = one.alyx.rest('files', 'list', django=fr_query)

repos = {repo['name']: ROOT_PATH_POPEYE.joinpath(repo['globus_path'][1:]) for repo in one.alyx.rest('data-repository', 'list') if repo['name'].startswith('flatiron')}

# %%
print(JOB_PREFIX)
for i, fr in enumerate(frs):
    local_path = repos[fr['data_repository']].joinpath(fr['relative_path']).parent
    print(f"{local_path}")
