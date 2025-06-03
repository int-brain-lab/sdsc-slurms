# Create new environment for trials extraction
project=trials-extraction
mkdir -p ~/Documents/PYTHON/$project
cd ~/Documents/PYTHON/$project
module purge
module load python/3.11
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/int-brain-lab/iblscripts.git
pip install -e iblscripts
git clone https://github.com/int-brain-lab/ibllib.git
pip install -e ibllib
git clone https://github.com/int-brain-lab/sdsc-slurms.git
# Add sdsc-slurms to path
export PATH=$PATH:~/Documents/PYTHON/sdsc-slurms
ONE_SAVE_ON_DELETE=0
SDSC_PATCH_PATH=/mnt/sdceph/users/ibl/data/quarantine/2025-03-03_trials-extraction/tasks