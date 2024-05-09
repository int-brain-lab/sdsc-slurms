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
'6d24683c-da42-4610-baf0-7ceee7014394',
'6e61f777-90b9-4656-af30-9ed54d426c5a',
'6eb8be8d-d089-43c6-b2e4-a6558ca16dcf',
'6efc58a4-e1cd-4eca-9205-7e4898cc1f8b',
'6fc4d73c-2071-43ec-a756-c6c6d8322c8b',
'701026df-e170-4ca7-88aa-eb0b95ef6ba1',
'727ff7c4-d63f-4bd8-9888-f134a25b874f',
'73ff4936-8cd3-4a75-a772-f563e67d249d',
'76de0e1a-30aa-4713-9fe5-25ad2dff653f',
'77121d92-6dde-4243-ab54-0a99efa22e99',
'7791ee46-5c13-4d1b-8518-5602dcb8666b',
'779fbed1-4b0e-4d7d-8882-6650690221a0',
'79628a45-c2e3-4206-97df-4c91edaff90f',
'79b2e46e-e2ee-446e-b875-9aff03ef52d2',
'79f44ba1-c931-4346-82eb-f628a9374045',
'79fbd14b-bfad-4b28-90a9-3019bf72336f',
'7a620688-66cb-44d3-b79b-ccac1c8ba23e',
'7b05cccc-44f6-4491-a0ea-e38d6e95513d',
'7bd5627e-b02e-47f1-b476-4bf8eaa726b3',
'7beb9419-113d-4e47-938c-68ab2657031e',
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