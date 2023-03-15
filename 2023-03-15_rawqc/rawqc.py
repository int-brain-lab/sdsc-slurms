import argparse
from pathlib import Path

import one.alf.files
from iblutil.util import setup_logger
from neurodsp.voltage import decompress_destripe_cbin
from brainbox.io.one import SpikeSortingLoader
_logger = setup_logger('ibllib')


PATH_SDSC_DATA = Path("/mnt/sdceph/users/ibl/data")

def make_jobs():
    from one.api import ONE
    pids = [
        "3d3d5a5e-df26-43ee-80b6-2d72d85668a5",
        "f2a098e7-a67e-4125-92d8-36fc6b606c45",
        "80f6ffdd-f692-450f-ab19-cd6d45bfd73e",
        "0aafb6f1-6c10-4886-8f03-543988e02d9e",
    ]

    one = ONE(base_url='https://alyx.internationalbrainlab.org')

    for pid in pids:
        sl = SpikeSortingLoader(one=one, pid=pid)
        cbin_file_relative = next(filter(lambda x: x.endswith('.ap.cbin'), sl.datasets))
        cbin_file = PATH_SDSC_DATA.joinpath(*sl.session_path.parts[-5:]).joinpath(cbin_file_relative)
        print(cbin_file)


def compute_ap(ap_file):
    # flac_folder = Path('/mnt/sdceph/users/ibl/data/SWC_NM_004/2021-10-29/001/raw_behavior_data')
    ap_file = Path(ap_file)
    session_path = one.alf.files.get_session_path(ap_file)
    output_file = Path('/mnt/home/owinter/ceph').joinpath(*session_path.parts[-5:]).joinpath(ap_file.relative_to(session_path))
    decompress_destripe_cbin(ap_file, output_file=output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run some metrics on microphone data')
    parser.add_argument('file_ap')
    parser.add_argument('out_path')
    args = parser.parse_args()
    compute_ap(args.file_ap)
