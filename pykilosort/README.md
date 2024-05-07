# Welcome to the pykilosort SDSC re-run guide

The purpose of this guide is to illustrate and document how to run pykilosort on SDSC and register the outputs to the Alyx datbase.

## Step 1: Installation

### Step 1.1: Setup the shared environment
Clone the SDSC slurms repository to your own home on SDSC.

```shell
cd ~/Documents/PYTHON
git clone git@github.com:int-brain-lab/sdsc-slurms.git
```

NB: There are instructions on how to install pykilosort environment, but for the scope if this May 2024 re-run,
the environment is already installed and hard-coded at this location: `/mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/python`

The task will look at a specific folder however, and it is necessary to setup iblscripts. Let's re-use Chris one for the time being
```shell
ln -s /mnt/home/clangfield/Documents/PYTHON/iblscripts ~/Documents/PYTHON/iblscripts
```

### Step 1.2: Create the output directory

The output task directory can only be private for the time being.
In the long term, we will create a shared permissions structure for IBL folks to access shared data.
For now, we have to deal with the fact that the task directory is user specific
```shell
mkdir /mnt/sdceph/users/ibl/data/quarantine/tasks_${USER}
chmod 775 -fR /mnt/sdceph/users/ibl/data/quarantine/tasks_${USER}/
```

### Step 1.3: Create directories for sbatch and log files

Create a directory under `/mnt/home/${USER}/Documents` containing a `logs` and `jobs` directory. For example,

```shell
mkdir -p /mnt/home/${USER}/Documents/spikesorting_rerun_2024/jobs
mkdir /mnt/home/${USER}/Documents/spikesorting_rerun_2024/logs
```

## Step 2: create the disbatch job files

Before doing anything, make sure you reserve your pids in the spreadsheet so that 
we don't overlap with each other.
https://docs.google.com/spreadsheets/d/1Cg7snZjPduzm-COgNrExkFoYDvPii5SURv_MwjF9wLA/edit#gid=175449612
Just put your name next to the pids you want to run

```shell
module load python/3.10.10
module load cuda/11.8.0
module load fftw/3.3.10
source /mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/activate
```

`sdsc-slurms/pykilosort/create_jobs.py` is a script that will create the job files for the pykilosort task.
Edit the `doc_path` at the top of this script to the path chosen in step 1.3. In the example you would set this to `/mnt/home/${USER}/Documents/spikesorting_rerun_2024/`. Then edit the pids in the scripts to match the pids you want to run.

The `${MAIL_USER}` field is an email address to which a notification will be sent if a slurm job fails. It would be advisable to change this from Chris's email address, but in any case make sure it is a valid email.

NB: there is a "copy me" column in the spreadsheet that you can use to copy the pids to the clipboard to paste into a Python list.


## Step 3: submit the jobs

```shell
sbatch /mnt/home/${USER}/Documents/spikesorting_rerun_2024/jobs/pykilosort_8b735d77-b77b-4243-8821-37802bf402fe.sbatch
sbatch ...
```

Look at the logs in `/mnt/home/${USER}/Documents/spikesorting_rerun_2024/logs`. You can monitor the status of submitted jobs with `squeue --me`.

## Abbreviated guide to slurm jobs

After a job is submitted, it will have one of 2 statuses visible with `squeue`: `R` (running) or `PD` (pending). A job remains pending due to either `(Priority)` (your jobs priority in the slurm dispatch queue) `(Resources)` (waiting for requested node and memory to become available) or `(QOSMaxCPUPerUserLimit)` (you have reached your limit for number of concurrent jobs). In any of these cases, the job will automatically be started once resources become available. 

To cancel a running slurm job (e.g. accidentally submitted the same job twice), find the slurm job id in the first column of the output of `squeue` and use

```shell
scancel {job id}
```

A range of job ids can be canceled via e.g. 

```shell
scancel {100..120}
```

This will cancel all of your submitted jobs with ids in the range 100 to 120. 

A job with `CG` status is in the process of exiting. It will generally remain in this status for a few seconds if the job has an error or is cancelled. 

Finally: **Do not submit slurm jobs programmatically. Flatiron Scientific Computing discourages this and will detect it.** Unfortunately the policy is they must be submitted manually. 



