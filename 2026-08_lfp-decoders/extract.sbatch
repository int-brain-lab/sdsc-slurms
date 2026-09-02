#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpdec"
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --time=23:59:59
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

# LFP feature extraction over the ephys-atlas cohort. Single node, joblib-parallel
# over PIDs (feature extraction is far cheaper per PID than the lfp-encoders
# regression fit, so no array striping is needed). Download the archives once first
# (racy across concurrent jobs, so do it on a login node before submitting):
#   python extract.py --download
# Then submit once per (--lfp-source, --snippet-mode) combination, e.g.:
#   sbatch extract.sbatch --lfp-source default --snippet-mode all
#   sbatch extract.sbatch --lfp-source aggressive --snippet-mode all
#   sbatch extract.sbatch --lfp-source uncompressed --snippet-mode all
#   sbatch extract.sbatch --lfp-source default --snippet-mode saturation-avoided
#   sbatch extract.sbatch --lfp-source aggressive --snippet-mode saturation-avoided
#   sbatch extract.sbatch --lfp-source uncompressed --snippet-mode saturation-avoided
# (the last three need snippets_avoiding_saturation.csv -- run saturation_snippets.py
# on a login node first, see README.md)

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-08_lfp-decoders/extract.py

module load python/3.11

# --workers 42 is the sbatch default (48-core node, ~10 GB/worker headroom); an
# explicit --workers in the passed args overrides it (argparse: last value wins).
$PYTHON_EXEC -u $PYTHON_SCRIPT --workers 42 "$@"  # e.g. --lfp-source default --snippet-mode all [--overwrite] [--limit N]
