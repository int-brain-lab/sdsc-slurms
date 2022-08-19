import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

import ibllib.io.extractors.training_audio as ta
from iblutil.util import get_logger

_logger = get_logger('ibl')


def check_sound(flac_folder, out_folder):
    # flac_folder = Path('/mnt/sdceph/users/ibl/data/SWC_NM_004/2021-10-29/001/raw_behavior_data')
    flac_folder = Path(flac_folder)
    out_folder = Path(out_folder)
    flac_file = next(flac_folder.glob('*.flac'))
    file_out = out_folder.joinpath('__'.join(Path(flac_file).parts[-5:-2]), 'welchogram.npz')
    _logger.info(f"welchogram output in {file_out}")
    wav, fs = sf.read(flac_file)
    tscale, fscale, W, detect = ta.welchogram(fs, wav, detect_kwargs=dict(threshold=.2))
    file_out.parent.mkdir(exist_ok=True)
    np.savez_compressed(file_out, W, tscale, fscale, detect)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run some metrics on microphone data')
    parser.add_argument('out_folder')
    parser.add_argument('flac_folder')
    args = parser.parse_args()
    check_sound(args.flac_folder, args.out_folder)

# local test:
# python check_microphone.py /datadisk/FlatIron/hausserlab/Subjects/PL015/2022-02-18/001/raw_behavior_data/_iblrig_micData.raw.flac /home/olivier/scratch/test

