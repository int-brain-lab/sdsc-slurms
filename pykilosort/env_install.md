Clone the pykilosort and dart repositories from github, they will be installed in dev mode
```shell
mkdir -p ~/Documents/PYTHON/pykilosort
cd ~/Documents/PYTHON/pykilosort
git clone git@github.com:int-brain-lab/pykilosort.git
git clone git@github.com:cwindolf/dartsort.git
git clone git@github.com:evarol/dredge.git
 ```

Create the enviornment and activate it
```shell
module load python/3.10.10
module load cuda/11.4.4
module load fftw/3.3.10
python -m venv ~/Documents/PYTHON/pykilosort/.venv
source ~/Documents/PYTHON/pykilosort/.venv/bin/activate
```

Install the requirements for each repo
```shell
cd ~/Documents/PYTHON/pykilosort
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd ../dredge
pip install -r requirements.txt
pip install -e .
cd ../dartsort
pip install -r requirements-full.txt
pip install -e .
```
