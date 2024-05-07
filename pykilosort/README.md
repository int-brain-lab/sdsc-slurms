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
```
ln -s /mnt/home/clangfield/Documents/PYTHON/iblscripts ~/Documents/PYTHON/iblscripts
```

### Step 1.2: Create the output directory

The output task directory can only be private for the time being.
In the long term, we will create a shared permissions structure for IBL folks to access shared data.
For now, we have to deal with the fact that the task directory is user specific
```
mkdir /mnt/sdceph/users/ibl/data/quarantine/tasks_${USER}
chmod 775 -fR /mnt/sdceph/users/ibl/data/quarantine/tasks_${USER}/
```

## Step 2: create the disbatch job files

Before doing anything, make sure you reserve your pids in the spreadsheet so that 
we don't overlap with each other.
https://docs.google.com/spreadsheets/d/1Cg7snZjPduzm-COgNrExkFoYDvPii5SURv_MwjF9wLA/edit#gid=175449612
Just put your name next to the pids you want to run

```
module load python/3.10.10
module load cuda/11.8.0
module load fftw/3.3.10
source /mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/activate
```

`sdsc-slurms/pykilosort/create_jobs.py` is a script that will create the job files for the pykilosort task.
Edit the pids in the scripts to match the pids you want to run.

NB: there is a "copy me" column in the spreadsheet that you can use to copy the pids to the clipboard to paste in a Python dict.

```shell

## Step 3: submit the jobs

```shell
sbatch /mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests/jobs/pykilosort_8b735d77-b77b-4243-8821-37802bf402fe.sbatch
sbatch ...
```

Look at the logs in `/mnt/home/clangfield/Documents/spikesorting_reruns_2024/benchmark_integration_tests/logs`
and you can monitor the queue with `squeue --me`.



