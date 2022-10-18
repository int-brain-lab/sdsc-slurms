```shell
# git clone https://github.com/int-brain-lab/pykilosort
module purge
module load python/3.8.12
module load cuda/11.4.4
module load fftw/3.3.10
python3 -m venv /mnt/home/owinter/Documents/PYTHON/envs/pykilosort
source /mnt/home/owinter/Documents/PYTHON/envs/pykilosort/bin/activate
cd ../pykilosort
pip install cupy-cuda11x click matplotlib mock numba numpy ibl-neuropixel phylib pydantic pytest scipy spikeinterface tqdm ibllib
pip install -e .
```