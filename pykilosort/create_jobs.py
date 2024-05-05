from pathlib import Path

sdsc_slurms_repo_path = Path.home().joinpath("Documents", "PYTHON", "sdsc-slurms")
doc_path = Path("/mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests")

# create jobs
slurm_template = sdsc_slurms_repo_path.joinpath("pykilosort", "disbatch_template.sbatch")
jobs_path = doc_path.joinpath("jobs")

pids = [
"8b735d77-b77b-4243-8821-37802bf402fe",
"8f2e16c4-893b-4f8f-bfb2-94fa452710ec",
]

for pid in pids:
    with open(slurm_template, "r") as template:
        fdata = template.read()
    fdata = fdata.replace("${PID}", pid)
    fdata = fdata.replace("${TIME}", "16:00:00")

    job_fn = jobs_path / f"pykilosort_{pid}.sbatch"

    with open(job_fn, "w") as jout:
        jout.write(fdata)
    print(f"sbatch {job_fn}")
