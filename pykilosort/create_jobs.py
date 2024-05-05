from pathlib import Path

sdsc_slurms_repo_path = Path.home().joinpath("Documents", "PYTHON", "sdsc-slurms")
doc_path = Path("/mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests")

# create jobs
slurm_template = sdsc_slurms_repo_path.joinpath("pykilosort", "disbatch_template.sbatch")
jobs_path = doc_path.joinpath("jobs")

pids = [
    "gnagna_1a276285-8b0e-4cc9-9f0a-a3a002978724",
    "gnagna_1e104bf4-7a24-4624-a5b2-c2c8289c0de7",
]

for pid in pids:
    with open(slurm_template, "r") as template:
        fdata = template.read()
    fdata = fdata.replace("${PID}", pid)
    fdata = fdata.replace("${TIME}", "16:00:00")

    job_fn = jobs_path / f"pykilosort_{pid}.sbatch"

    with open(job_fn, "w") as jout:
        jout.write(fdata)
