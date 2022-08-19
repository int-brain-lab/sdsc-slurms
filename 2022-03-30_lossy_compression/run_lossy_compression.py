import argparse
from pathlib import Path

from mtscomp.lossy import compress_lossy

# Parse input arguments
parser = argparse.ArgumentParser(description='Run destripe')
parser.add_argument('out_dir')
parser.add_argument('in_dir')
args = parser.parse_args()

out_dir = Path(str(args.out_dir))
in_dir = Path(str(args.in_dir))

# Construct out_file and create directory
cbin_file = next(in_dir.glob('*ap*.cbin')) 
cmeta_file = next(in_dir.glob('.'.join(cbin_file.name.split('.')[:-2]) + '*.ch'))

probe_dir = out_dir.joinpath(cbin_file.parent.relative_to("/mnt/sdceph/users/ibl/data"))
probe_dir.mkdir(exist_ok=True, parents=True)

out_lossy = probe_dir.joinpath(Path(cbin_file.name).with_suffix('.lossy.npy'))
out_svd = probe_dir.joinpath(Path(cbin_file.name).with_suffix('.svd.npz'))

print(cbin_file)
print(cmeta_file)
print()
print(out_lossy)
print(out_svd)
# Run function
#res = compress_lossy(cbin_file, cmeta=cmeta_file, rank=80, overwrite=True, out_lossy=out_lossy, out_svd=out_svd)
