#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpenc"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --array=0
#SBATCH --time=1-00:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-07_lfp-encoders/_log_.%x_%A_%a.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-07_lfp-encoders/_log_.%x_%A_%a.err

# Brain-wide LFP-encoding fit. Behaviour is loaded from ONE (OneSdsc, local mirror
# -- no bwm_behavior shards, so wheel is complete). The compressed tiers read from a
# consolidated lfpack archive downloaded from S3; the uncompressed tier reads per-PID
# Cadzow checkpoints. Download the archive(s) once first (racy across array tasks):
#   python encode.py --download --lfp-source default
#   python encode.py --download --lfp-source aggressive
# Then run three times to compare tiers (pass one --lfp-source each):
#   sbatch encode.sbatch --lfp-source default
#   sbatch encode.sbatch --lfp-source aggressive
#   sbatch encode.sbatch --lfp-source uncompressed
# The --array stripes that source's PIDs across nodes (task i takes PIDs i::array_count).

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-07_lfp-encoders/encode.py

module load python/3.11

# --workers 42 is the sbatch default (48-core node, ~6 GB/worker); an explicit
# --workers in the passed args overrides it (argparse: last value wins).
$PYTHON_EXEC -u $PYTHON_SCRIPT --workers 42 "$@"  # e.g. --lfp-source aggressive [--overwrite] [--limit N]