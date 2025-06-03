from pathlib import Path
from itertools import groupby
import pickle
from ibllib.oneibl.patcher import SDSCPatcher
from one.api import ONE
from packaging import version
from ibllib import __version__
from one.alf.files import get_session_path
import shutil
from datetime import datetime


DATASETS = ('_ibl_trials.table.pqt', '_ibl_trials.stimOff_times.npy',
            '_ibl_trials.stimOnTrigger_times.npy', '_ibl_trials.stimOffTrigger_times.npy')

def correct_version(v):
    try:
        return version.parse(v or '') >= version.Version('2.38.0')
    except version.InvalidVersion:
        return False

assert correct_version(__version__)

QUAR_PATH = Path("/mnt/ibl/quarantine")
TMP = Path("/home/datauser/temp")

with open(QUAR_PATH.joinpath("2024-07-15_paths.pkl"), "rb") as fp:
    processed_paths = pickle.load(fp)
with open(QUAR_PATH.joinpath("2024-07-15_stim-times.pkl"), "rb") as fp:
    processed = pickle.load(fp)

one = ONE(cache_rest=None)

sdsc_patcher = SDSCPatcher(one=one)

def check_skip(eid, one):
    dsets = one.alyx.rest('datasets', 'list', session=eid, django='name__startswith,_ibl_trials')
    dsets = [d for d in dsets if d['name'] in DATASETS]
    skip = True
    for name, dd in groupby(dsets, lambda d: f'{d["collection"]}/{d["name"]}'):
        skip &= any(
            correct_version(d['version'])
            if d['version']
            else
            datetime.fromisoformat(d['created_datetime']) > datetime(2024, 7, 10)
            for d in dd
        )
    return skip

def move_to_revision_dir(paths, revision='2024-07-15'):
    base_dir = paths[0].parent
    # A lot appear to have already been moved so if the folder already exists,
    # simply return those paths
    rev_dir = base_dir.joinpath(f'#{revision}#')
    new_paths = [rev_dir.joinpath(p.name) for p in paths]
    if rev_dir.exists():
        if all(map(Path.exists, new_paths)):
            return new_paths
        else:  # just start again (this may lead to file not found error)
            shutil.rmtree(rev_dir)
    try:
        rev_dir.mkdir()
    except PermissionError:
        print("Couldn't register:")
        print(paths)
        print('Add datauser permissions on Popeye')
        return
    new_paths = []
    for p in paths:
        new_paths.append(p.rename(rev_dir.joinpath(p.name)))
    return new_paths

re_extract = []
for eid, paths in processed_paths.items():
    if not all(map(Path.exists, paths)):
        print(f"Re-extract required: {eid}, {paths[0].parents[1]}")
        re_extract.append(eid)
for eid in re_extract:
    del processed_paths[eid]

with open(QUAR_PATH.joinpath("2024-07-15_paths.pkl"), "wb") as fp:
    pickle.dump(processed_paths, fp)
with open(QUAR_PATH.joinpath("2024-07-15_paths.pkl"), "rb") as fp:
    processed_paths = pickle.load(fp)

print('Warning | the following sessions were not fully extracted:')
print(re_extract)

to_remove = []
for eid, paths in processed_paths.items():
    print(eid)
    if not paths:
        print(f"Problem with {eid}, check 2024-07-15_stim-times.pkl for possible error")
        continue

    if check_skip(eid, one):
        print(f"{eid} already done")
        for sess_path in set(map(get_session_path, paths)):
            if sess_path.exists():
                print(f'Clearing {sess_path}')
                shutil.rmtree(sess_path, ignore_errors=True)
        to_remove.append(eid)
        if eid not in processed:
            processed[eid] = ()
        continue
    
    # The paths are saved by task_name/session_path meaning session paths not unique
    for _, pp in groupby(paths, lambda x: get_session_path(x)):
        pp = list(pp)
        try:
            responses = sdsc_patcher.patch_datasets(pp, dry=False, versions=__version__)
        except FileExistsError as e:
            if 'Protected datasets were found in the file list.' in str(e):
                print('Protected; forcing revision')
                new_paths = move_to_revision_dir(pp)
                if new_paths is None:
                    continue  # Permissions error
                responses = sdsc_patcher.patch_datasets(new_paths, dry=False, versions=__version__, revision='2024-07-15', force=True)
            else:
                print(e)
                continue
        except Exception as e:
            print(f'Error | {e}')
            continue


for eid in to_remove:
    del processed_paths[eid]

with open(QUAR_PATH.joinpath("2024-07-15_stim-times.pkl"), "wb") as fp:
    pickle.dump(processed, fp)

with open(QUAR_PATH.joinpath("2024-07-15_paths.pkl"), "wb") as fp:
    pickle.dump(processed_paths, fp)