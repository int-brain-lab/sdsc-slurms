from pathlib import Path

sdsc_slurms_repo_path = Path.home().joinpath("Documents", "PYTHON", "sdsc-slurms")
doc_path = Path("/mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests")

# create jobs
slurm_template = sdsc_slurms_repo_path.joinpath("pykilosort", "disbatch_template.sbatch")
jobs_path = doc_path.joinpath("jobs")

substitutions = {
    'LOG_PATH': str(doc_path.joinpath("logs")),
    'TIME': "16:00:00",
    'PID': None,
    'MAIL_USER': 'olivier.winter@internationalbrainlab.org'
}

pids = [
    '00a824c0-e060-495f-9ebc-79c82fef4c67',
]

OVERWRITE = False
for pid in pids:
    substitutions['PID'] = pid
    job_fn = jobs_path / f"pykilosort_{pid}.sbatch"
    if job_fn.exists():
        if not OVERWRITE:
            print(f"Job file already exists: {job_fn}")
            continue
    with open(slurm_template, "r") as template:
        fdata = template.read()
    for k, v in substitutions.items():
        fdata = fdata.replace(f"${{{k}}}", v)

    with open(job_fn, "w") as jout:
        jout.write(fdata)
    print(f"sbatch {job_fn}")
