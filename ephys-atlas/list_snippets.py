"""
Check the ceph drive for complete jobs
"""
from functools import reduce
from pathlib import Path

import scipy.signal
import pandas as pd

from one.api import ONE


root_path = Path('/mnt/home/owinter/ceph/EA')
subjects = sorted([p.name for p in root_path.glob('*') if p.is_dir()])


n_snippets = 0
for i, subject in enumerate(subjects):
    snippet_paths = [sn.parent for sn in root_path.joinpath(subject).rglob('wav.pqt')]
    n_snippets += len(snippet_paths)


print("number of snippets: ", n_snippets)