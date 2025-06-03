"""How long does a single trials extraction task take on average?

Executable: ChoiceWorldTrialsNidq
M: 38.98 seconds, Median: 32.49 seconds, n=127
Executable: ChoiceWorldTrialsBpod
M: 31.69 seconds, Median: 27.12 seconds, n=1501

"""
import re
import numpy as np
from one.webclient import AlyxClient

pattern = re.compile('--- (?P<t>\d+\.\d+) seconds run-time ---')
alyx = AlyxClient()
for executable in ('ChoiceWorldTrialsNidq', 'ChoiceWorldTrialsBpod'):
    print(f'Executable: {executable}')
    # Fetch tasks that match the executable and have a log entry with the pattern
    tasks = alyx.rest('tasks', 'list', status='Complete', executable=f'ibllib.pipes.behavior_tasks.{executable}', django='name__startswith,Trials_')
    # matches = filter(None, map(pattern.search, (task['log'] or '' for task in tasks)))
    # durations = np.fromiter(map(lambda m: float(m.groups()[0]), matches), dtype=float)
    durations = np.fromiter(filter(None, (task['time_elapsed_secs'] for task in tasks)), dtype=float)
    # Print the mean and median durations
    print(f'M: {durations.mean():.2f} seconds, Median: {np.median(durations):.2f} seconds, n={len(durations)}')
