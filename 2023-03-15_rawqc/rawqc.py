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
        cbin_file = next(cbin_file.parent.glob('.ap.*.cbin'))
        print(cbin_file, cbin_file.exists())


def compute_ap(ap_file):
    # flac_folder = Path('/mnt/sdceph/users/ibl/data/SWC_NM_004/2021-10-29/001/raw_behavior_data')
    ap_file = Path(ap_file)
    session_path = one.alf.files.get_session_path(ap_file)
    output_folder = Path('/mnt/home/owinter/ceph').joinpath(*session_path.parts[-5:]).joinpath(ap_file.relative_to(session_path)).parent
    output_folder.mkdir(parents=True, exist_ok=True)
    output_folder.joinpath('raw.ap.cbin')
    meta_file = output_folder.joinpath('raw.ap.meta')
    for ext in ('ch', 'cbin', 'meta'):
        sdsc_file = next(ap_file.parent.glob(f'*.ap.*.{ext}'))
        symlink = output_folder.joinpath(f'raw.ap.{ext}') 
        if not symlink.is_symlink():
            symlink.symlink_to(sdsc_file)
    output_file = output_folder.joinpath('destriped_file.bin')
    ap_link = output_folder.joinpath('raw.ap.cbin')
    decompress_destripe_cbin(ap_link, output_file=output_file, nprocesses=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run some metrics on microphone data')
    parser.add_argument('file_ap')
    args = parser.parse_args()
    compute_ap(args.file_ap)
