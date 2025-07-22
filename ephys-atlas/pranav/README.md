# Steps to run the feature computations and aggregations using disBatch.
## Example for the psychedelics dataset

# Step 1
Run [extract_pyschedelics_pids.py](./extract_pyschedelics_pids.py) file.
Creates a CSV file with pid, eid and probe_name.

# Step 2
Run [create_snippets_file.py](./create_snippets_file.py) file.
Creates a CSV file with Snippets information - t_starts and durations.

# Step 3
Create the [computation.py](./computation.py) file.
Runs the computation for one snippet. This script will be launched in parallel by disBatch.
### Note
In case the parallelization needs to be done via joblib or some other mechanisms, the above script needs to be parallelized accordingly.

# Step 4
Create the [Runprogram.sh](./Runprogram.sh) file.
Shell script to launch the computation, activate venv, and handle stdin, stdout for the computation.

# Step 5
Create the Task file using [create_task_file.py](./create_task_file.py)
This is utility script to create the task file that is used by disBatch to launch all the commands.
Each line is the exact command that each task will run on SDSC.

# Launch the Task file using disBatch
```bash
module load disBatch
sbatch -n 96 -c 1 -p gen -t 23:59:59 disBatch -p v1_run --fill Full_Task_file
```

# Aggregation.
Run the aggregation using [aggregation.py](./aggregation.py)
