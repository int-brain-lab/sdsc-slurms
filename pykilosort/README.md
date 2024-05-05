# Welcome to the pykilosort SDSC re-run guide

The purpose of this guide is to illustrate and document how to run pykilosort on SDSC and register the outputs to the Alyx datbase.

## Step 1: Installation
Clone the SDSC slurms repository to your drive.

```shell
cd ~/Documents/PYTHON
git clone git@github.com:int-brain-lab/sdsc-slurms.git
```

There are instructions on how to install pykilosort environment, but for the scope if this May 2024 re-run,
the environment is already installed and hard-coded at this location:
`PYTHON_EXEC=/mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/python`

## Step 2: create the disbatch job files
```
source/mnt/home/clangfield/Documents/PYTHON/envs/pyks2/bin/python
```