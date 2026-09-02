#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpdec_agg"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --time=4:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

# Aggregate extracted snippet features into per-combo tables (aggregate.py). Run as
# its own sbatch job, not on the login node: ephysatlas.aggregation.
# produce_output_dataframes hardcodes joblib Parallel(n_jobs=-1), which spawns one
# worker per visible CPU -- on the login node that blows past the per-session
# memory cgroup cap (observed OOM-kill even for a single combo over the full
# ~1099-PID cohort) even though the node itself has memory free. On a dedicated
# compute node the -1 workers only see this job's own allocated cores/memory.
#   sbatch aggregate.sbatch --lfp-source default --snippet-mode all

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-08_lfp-decoders/aggregate.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT "$@"  # e.g. --lfp-source default --snippet-mode all
