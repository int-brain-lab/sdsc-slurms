"""Get the list of sessions that require trials table re-extraction.

The first step (get_sessions) identifies sessions that have trials and are either older than the
date of the fixes, or have the revised datasets.

The second step (check_skip) checks if the session datasets are already up-to-date based on the
created date. This is required because some sessions will have already been re-extracted without
the revision.

Run on SDSC as datauser within the Django shell:
>>> cd ~/Documents/PYTHON/alyx/
>>> source alyxvenv/bin/activate
>>> python alyx/manage.py shell
"""
import re
from pathlib import Path
from itertools import filterfalse
from functools import partial
from uuid import UUID
import pickle
from packaging import version
from one.api import ONE
from ibllib import __version__
import pandas as pd
from datetime import datetime, date
from data.models import Dataset
from .constants import REVISION_BPOD, REVISION_FPGA, VERSION_BPOD, VERSION_FPGA, ROOT, DATASETS, correct_version

assert correct_version(__version__, VERSION_FPGA), 'ibllib version is too old, please update'
assert version.parse(ONE.version) >= version.Version('3.1.1'), 'ONE version is too old, please update'

QUAR_PATH = Path('/mnt/ibl/quarantine')
RUN_LIST = QUAR_PATH.joinpath(f'{REVISION_FPGA}_trials-extraction', f'{REVISION_FPGA}_run_list.pkl')

one = ONE(cache_rest=None)
one.load_cache()

if (processed_file := QUAR_PATH.joinpath(f'{REVISION_FPGA}_processed.pkl')).exists():
    with open(processed_file, 'rb') as fp:
        processed = pickle.load(fp)
else:
    processed = {}


def has_all_revised_datasets(s, revision=REVISION_BPOD):
    """Check if the session has all required datasets for the specified revision."""
    required_attributes = [x.split('.')[1] for x in DATASETS]
    pattern = re.compile(
        fr'alf(/task_\d{2})?/#{revision}#/(?P<ds>{"|".join(map(re.escape, DATASETS))})$')
    return sum(s.str.match(pattern)) == len(required_attributes)


def get_sessions(one, unprocessed_only=True):
    """There are different ways to determine whether a session needs re-extraction:
    
    1. If the session is Bpod-only, is it newer than the first revision date or does it have all
       the revised datasets?
    
    2. If the session is FPGA, check whether it was processed after the most recent revision date
       or does it have all the newely revised datasets?
    
    Sessions with no trials datasets are excluded also.
    """
    df = one._cache.datasets
    has_trials = df['rel_path'].groupby('eid').agg(lambda x: any(x.str.contains('_ibl_trials.')))
    bpod_only = df['rel_path'].groupby('eid').agg(lambda x: not any(x.str.contains('raw_sync_data|raw_ephys_data')))
    with_old_revision = df['rel_path'].groupby('eid').agg(has_all_revised_datasets)

    # First exclude Bpod-only sessions that are newer than the revision date or have the revised datasets
    post_revision = one._cache.sessions['date'] > date.fromisoformat(REVISION_BPOD)
    ok_bpod = bpod_only & (post_revision | with_old_revision)

    # Now exclude FPGA sessions that are newer than the revision date or have the revised datasets
    with_new_revision = df['rel_path'].groupby('eid').agg(partial(has_all_revised_datasets, revision=REVISION_FPGA))
    post_revision = one._cache.sessions['date'] > date.fromisoformat(REVISION_FPGA)
    ok_fpga = ~bpod_only & (post_revision | with_new_revision)

    # Combine both conditions to get the sessions that need re-extraction
    todo = has_trials & ~(ok_bpod | ok_fpga)

    eids = todo[todo].index
    fcn = lambda x: ROOT.joinpath(x['lab'], 'Subjects', x['subject'], str(x['date']), str(x['number']).zfill(3))
    session_paths = one._cache.sessions.loc[eids].apply(fcn, axis=1)
    if unprocessed_only:
        return filter(lambda x: x[0] not in processed, session_paths.items())
    return session_paths.items()


def check_skip(session_dsets):
    """Check if the session datasets have already been re-extracted based on the created date."""
    def is_new(d):
        if d['version']:
            return correct_version(d['version'], VERSION_BPOD if d['bpod_only'] else VERSION_FPGA)
        return d['created_datetime'] > datetime.fromisoformat(REVISION_BPOD if d['bpod_only'] else REVISION_FPGA)
    # First check Bpod-only sessions
    session_dsets = session_dsets[session_dsets['name'].map(lambda x: x in DATASETS)]
    session_dsets = session_dsets.set_index(['collection', 'name'])
    if len(session_dsets.index.get_level_values(1).unique()) != len(DATASETS):
        return False
    session_dsets['is_new'] = session_dsets.apply(is_new, axis='columns')
    return session_dsets.groupby(level=[0, 1])['is_new'].apply(any).all()


if RUN_LIST.exists():
    with open(RUN_LIST, 'rb') as fp:
        eids, session_paths = zip(*pickle.load(fp))
else:
    eids, session_paths = zip(*get_sessions(one))


dsets = Dataset.objects.filter(session__in=eids, name__startswith='_ibl_trials').values_list('session', 'collection', 'name', 'version', 'created_datetime')
df = pd.DataFrame(dsets, columns=('eid', 'collection', 'name', 'version', 'created_datetime')).set_index('eid')
# add bpod-only column as this affects the date and version check
df['bpod_only'] = one._cache.datasets['rel_path'].groupby('eid').agg(lambda x: not any(x.str.contains('raw_sync_data|raw_ephys_data')))
skip = df.groupby('eid').apply(check_skip)
run_list = list(filterfalse(lambda x: skip.get(x[0]), zip(eids, session_paths)))

if not RUN_LIST.parent.exists():
    RUN_LIST.parent.mkdir(parents=True, exist_ok=True)
with open(RUN_LIST, 'wb') as fp:
    pickle.dump(run_list, fp)
