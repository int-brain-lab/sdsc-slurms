#!/usr/bin/bash
#SBATCH -p gen
#SBATCH --job-name="lfpack-publish"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH -o /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%j.out
#SBATCH -e /mnt/home/owinter/Documents/sdsc-slurms/2026-06-lfpack/_log_.%x_%j.err

# Upload the labelled merged archives of a release straight from ceph to S3 (ibl profile).
# Run after attach.sbatch (sbatch --dependency=afterok:<attach job> publish.sbatch).
# Pass --dryrun first to list what would be uploaded.
#   BWM (public):          ibl-brain-wide-map-public/resources/ibl-agent-data
#   ephys-atlas (private): ibl-brain-wide-map-private/resources/lfp/ephys-atlas

VERSION=v05
OUTPUT_ROOT=/mnt/home/owinter/ceph/ea/denoised_lfp
AWS="$HOME/.local/bin/aws --profile ibl"

set -e
$AWS s3 sync $OUTPUT_ROOT s3://ibl-brain-wide-map-public/resources/ibl-agent-data \
  --exclude '*' --include "lf_compressed_${VERSION}_*_all_bwm.h5" --no-progress "$@"
$AWS s3 sync $OUTPUT_ROOT s3://ibl-brain-wide-map-private/resources/lfp/ephys-atlas \
  --exclude '*' --include "lf_compressed_${VERSION}_*_all.h5" --no-progress "$@"
