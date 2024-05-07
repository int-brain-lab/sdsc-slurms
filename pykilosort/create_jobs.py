from pathlib import Path

sdsc_slurms_repo_path = Path.home().joinpath("Documents", "PYTHON", "sdsc-slurms")
doc_path = Path("/mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests")

# create jobs
slurm_template = sdsc_slurms_repo_path.joinpath("pykilosort", "disbatch_template.sbatch")
jobs_path = doc_path.joinpath("jobs")

substitutions = {
    'LOG_PATH': doc_path.joinpath("logs"),
    'TIME': "16:00:00",
    'PID': None,

}
pids = [
    'bf827e50-eed3-4984-908e-c48ffbeae7b5',
    'c07d13ed-e387-4457-8e33-1d16aed3fd92',
    'c0c3c95d-43c3-4e30-9ce7-0519d0473911',
    'c16b8e98-100f-40ce-b417-b1524c7e1270',
    'c4f6665f-8be5-476b-a6e8-d81eeae9279d',
    'c5b9e063-f640-4936-b851-f7602cb6659b',
    'c6ba6f8e-c13e-410f-b7df-e193ba0d239d',
]



for pid in pids:
    substitutions['PID'] = pid
    job_fn = jobs_path / f"pykilosort_{pid}.sbatch"
    # if job_fn.exists():
    #     print(f"Job file already exists: {job_fn}")
    #     continue
    with open(slurm_template, "r") as template:
        fdata = template.read()
    for k, v in substitutions.items():
        fdata = fdata.replace(f"${{{k}}}", v)

    with open(job_fn, "w") as jout:
        jout.write(fdata)
    print(f"sbatch {job_fn}")
