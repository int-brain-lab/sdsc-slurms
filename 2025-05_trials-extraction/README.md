# IBL choice world trials re-extraction


--------------------
Revision: 2025-03-03
Version: 3.3.0
--------------------
This version includes the fix for incorrect first trial extraction in some FPGA sessions.
https://github.com/int-brain-lab/ibllib/issues/909
https://github.com/int-brain-lab/ibllib/pull/943 - release

--------------------
Revision: 2024-07-15
Version: 2.38.0
--------------------
This version extracts stimulus times by taking first TTL time within a fixed window after the Bpod
trigger.  At this time, the Bpod trigger times were also saved by default.
https://github.com/int-brain-lab/iblrig/issues/654
https://github.com/int-brain-lab/ibllib/issues/775
https://github.com/int-brain-lab/ibllib/pull/788 - feature branch
https://github.com/int-brain-lab/ibllib/pull/802 - release


## Files & paths

### quarentine/2025-03-03_trials-extraction/2025-03-03_processed.pkl
A dict of eid -> tuple of exceptions (empty is complete)

### quarentine/2025-03-03_trials-extraction/2025-03-03_run_list.pkl
A list of tuples containing (eid, session_path). The session path is via the mounted drive accessible from Popeye.
