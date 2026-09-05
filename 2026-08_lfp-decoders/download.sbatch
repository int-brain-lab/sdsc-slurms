#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpdec_dl"
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-08_lfp-decoders/_log_.%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=olivier.winter@internationalbrainlab.org

# Download the region-annotated compressed archives (extract.py --download). Run as
# its own sbatch job, not on the login node: boto3's multipart download opens several
# concurrent buffered file handles, which can hit the per-session memory cgroup cap
# on a login node (observed `OSError: [Errno 12] Cannot allocate memory` from a
# transfer-thread file seek) even though the node itself has memory free.
#   sbatch download.sbatch --lfp-source mild --download

PYTHON_EXEC=/mnt/home/owinter/Documents/ephys-atlas/.venv/bin/python
PYTHON_SCRIPT=~/Documents/sdsc-slurms/2026-08_lfp-decoders/extract.py

module load python/3.11

$PYTHON_EXEC -u $PYTHON_SCRIPT "$@"  # e.g. --lfp-source mild --download
