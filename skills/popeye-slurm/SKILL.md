---
name: popeye-slurm
description: Use when writing, adapting, or submitting Slurm jobs on the Flatiron Institute Popeye cluster (sbatch scripts, partitions, constraints, node types). Restricted to Popeye — we have no Rusty access.
---

# Submitting Slurm jobs on Popeye

Reference: "Slurm Partitions and Constraints" (SCC docs, version 29, 08 Apr 2026). This skill only covers **Popeye** — ignore any Rusty-only rows/hardware in that doc (e.g. `gpuxl`/`gpup`, `rome`/`genoa`/`graniterapids` nodes, the h100/h200/rtxblackwell/a100-80gb GPU nodes all live on Rusty, not Popeye).

## No SSH access

Claude cannot log into Popeye — the login is MFA-protected and cannot be completed from here. Never attempt `ssh`. Instead:

1. Write/edit the `.sbatch` file and its Python script locally in this repo (`sdsc-slurms/`).
2. Tell the user the exact command to run themselves on Popeye, e.g.:
   ```
   cd ~/Documents/sdsc-slurms/<job-dir> && sbatch <name>.sbatch [extra args]
   ```
3. If the user reports back `squeue`/log output, help interpret it, but do not claim to have submitted or checked anything yourself.

## Job layout convention

Each job lives in its own folder at the top of this repo, named `yyyy-mm_short-name` (or `yyyy-mm-dd_short-name` for a one-off). A folder typically contains:

- `<name>.py` — the actual work, run with `python -u`, argparse-based so extra flags can be passed through from sbatch
- `<name>.sbatch` — the submission script (see template below)
- optionally `README.md` if the job needs explaining beyond the script itself

On the cluster, the home directory is `/mnt/home/owinter/Documents/`, and jobs use the project venv at `/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python` after `module load python/3.11`. Log files are named `_log_.%x_%j.out` (or `_log_.%x_%A_%a.out` for array jobs) and are gitignored — never remove that ignore rule.

## Standard sbatch template

```bash
#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="<short-name>"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/<job-dir>/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/<job-dir>/_log_.%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/<job-dir>/<name>.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT "$@"
```

Adapt per job:
- Add `#SBATCH --mem=500G` (or similar) for memory-heavy single-node jobs.
- Add `#SBATCH --array=0-N` and use `%A_%a` in the log path for array jobs (task index via `$SLURM_ARRAY_TASK_ID`).
- Drop `-e` if a single combined `.out` log is fine (many existing jobs do this).
- Use a per-job `$SCRATCH_ROOT=/scratch/<name>_${SLURM_JOB_ID}` directory for transient large intermediates, and `rm -rf` it at the end of the script.

## Partitions on Popeye

| Partition | Use | Limit | Time (default / max) | Exclusive node? | Flags |
|---|---|---|---|---|---|
| General | default | 4 nodes or 256 cores | 1 day / 1 week | Yes | `-p gen` |
| Express serial | small serial jobs, can share a node | 256 cores aggregate | 6h / 1 week | No | `-p genx -c <N>` |
| Big memory | single-node, needs 1.5–3TB RAM | 2 nodes | — / 1 week | Yes | `-p mem` |
| Preemptive | large jobs, must checkpoint/restart | no limit | — / 1 week | Yes | `-p preempt --qos=preempt` |
| Request | very large, needs manual OK | — | — / 1 week | Yes | `-p request --qos=request` (then email scicomp@flatironinstitute.org) |
| GPU | needs a GPU | 32 GPUs/user | — / 1 week | No | `-p gpu --gpus=<N> -c <M>` |
| GPU preempt | preemptable GPU | no limit | — / 1 week | No | `-p gpupreempt -q gpupreempt` |

Default to `-p gen` unless the job specifically needs big memory, a GPU, or is preemptible/very long-running. `gen` allocates the whole node even for one core, so size `--cpus-per-task` to the node type (48 for skylake/cascadelake, 64 for icelake) rather than requesting partial cores.

## Node types / constraints on Popeye

| #Nodes | CPU | Cores | Memory | GPU | Flag |
|---|---|---|---|---|---|
| 144 | skylake | 48 | 768GB | — | `-C skylake` |
| 432 | cascadelake | 48 | 768GB | — | `-C cascadelake` |
| 216 | icelake | 64 | 1TB | — | `-C icelake` |
| 1 | cooperlake | 96 | 3TB | — | `-p mem` |
| 4 | skylake | 48 | 768GB | 4x V100-32GB | `-C v100` |
| 1 | — | 64 | 1TB | 6x A100-40GB | `-C a100` |

For MPI/high-throughput-fabric jobs, add `-C ib` for InfiniBand.

## GPU jobs

```
-p gpu --gpus=<N> -c <M> [-C v100|a100]
```
- `N` = GPUs needed, `M` = CPU cores (at least 1 per GPU); add `--mem=<...>` if the default per-GPU memory isn't enough.
- Nodes are shared between users by GPU count — **never add `--exclusive`**.
- `-p gpupreempt -q gpupreempt` goes past the per-user GPU limits but can be killed by regular `gpu` jobs at any time.

## Checking availability and usage before submitting

```
module load fi-utils
fi-nodes            # current free CPUs/nodes/GPUs
fi-slurm-limits      # your usage vs. partition limits
```
`https://grafana.flatironinstitute.org` has historical cluster load if `fi-nodes` isn't enough context.

## Many small tasks: disBatch

For large numbers of short/serial tasks that shouldn't each get a whole node, this repo has used disBatch to pack many tasks onto one `gen` allocation (see `2022-08-19_microphone_ephys/`, `2023-03-15_rawqc/`, `pykilosort/` for examples of the `.disbatch` task-list + wrapper `.sbatch` pattern) rather than one `sbatch` submission per task.

## Workflow when asked to set up/submit a job

1. Find the relevant existing job folder, or create a new `sdsc-slurms/yyyy-mm_short-name/` one.
2. Write/adapt the Python script and the `.sbatch` file using the template and partition/constraint tables above, picking resources to match the job's actual needs.
3. Give the user the exact `sbatch` command to run themselves — do not attempt to run or check it via SSH.
