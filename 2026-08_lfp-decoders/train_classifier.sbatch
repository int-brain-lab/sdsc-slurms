#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpdec_train"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --time=4:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

# Train the 5-fold XGBoost region classifier (train_classifier.py). Run as its own
# sbatch job, not on the login node -- same login-node memory-cgroup risk as
# aggregate.sbatch, now over the full ~1099-PID cohort's worth of channels
# (the 20-PID smoke test fit on the login node fine; this is ~55x more channels).
#   sbatch train_classifier.sbatch --lfp-source default --snippet-mode all

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-08_lfp-decoders/train_classifier.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT "$@"  # e.g. --lfp-source default --snippet-mode all
