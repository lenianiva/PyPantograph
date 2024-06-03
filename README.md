# PyPantograph

Python interface to the Pantograph library

## Getting started
<!-- Update submodule
``` bash
git submodule update --init
```
Install dependencies
```bash
poetry install
``` -->

<!-- First initialize the git submodules so that git can keep track of the submodules being used do:
```bash
# - initialize the git submodules by preparing the git repository, but it does not clone or fetch them, just init's git's internal configs
git submodule init
```
Then to clone, fetch & update the submodules code (and also initilize anything you might have forgotten that is specificed in the `.gitmodules` file):
```bash
# - initialize the git submodules so that git can track them and then the update clone/fetches & updates the submodules
git submodule update --init
```

Then install poetry by (e.g., [by following poetry's official instructions](https://python-poetry.org/docs/#installing-manually)).

Then once you confirm you have poetry & the initialized git submodules, execute:
```bash
poetry build
```
To run server tests:
``` bash
python -m pantograph.server
```
The tests in `pantograph/server.py` also serve as simple interaction examples -->

## Install 1: With Conda and Pip in the SNAP cluster

```bash
# - Install Lean4 manually (elan and lake), 1st one is the SNAP one, 2nd is the most common one
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
# curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s

# - Make sure Lean4 tools (lean, lake) are available 
export PATH="$HOME/.elan/bin:$PATH"
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
bash
elan
lake

# - Create and activate the right python env (this is needed so that poetry build works)
conda create -n pypantograph_env python=3.11 -y
conda activate pypantograph_env
#conda remove --name pypantograph_env --all

# - Install poetry with python venv (needs seperate install so poetry & your projs deps don't crash)
mkdir $HOME/.virtualenvs

# put the follow BEFORE your conda init stuff in your .bashrc
export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry
export PATH="$VENV_PATH/bin:$PATH"

# now actually install poetry in a python env after creating an python env for poetry with venv
python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry

poetry

# - Init the git submodules (i.e., make git aware of them/track them) + fetch/clone/update (and double check submodule is inited)
git submodule init
git submodule update --init

# - For snap make sure the repo is sym linked you're using your
git clone git@github.com:lenianiva/PyPantograph.git
git checkout <your-branch>
ln -s $AFS/PyPantograph $HOME/PyPantograph

# - Build the PyPantograph proj (build the py distribution, py deps and custom (lean4) installs). Note: pip install -e doesn't work on the dist .whl builds etc so you instead the next command
cd $HOME/PyPantograph
poetry build

# - Install pypantograph in editable mode (only pyproject.toml (or setup.py!) needed! Assuming your at the proj root)
cd $HOME/PyPantograph
pip install -e . 

# - Confirm intalls
pip list | grep pantograph
pip list | grep vllm
pip list | grep torch

# - Select freeiest GPU wrt vRAM
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
echo $CUDA_VISIBLE_DEVICES

# - Make sure the PyPantrograph server tests by Leni work
cd ~/PyPantograph
# python -m pantograph.server
python $HOME/PyPantograph/pantograph/server.py
python $HOME/PyPantograph/test_vllm.py
```
Note: the tests in `pantograph/server.py` also serve as simple interaction examples

References:
- My SNAP `.bashrc`: https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc
    - Especially useful for Conda vs Poetry export order
- Poetry in SNAP: https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#poetry
- Gitsubmodules: https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#git-submodules
- Lean in SNAP: https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#lean-in-snap
- ChatGPT: https://chat.openai.com/c/e01336a7-6f67-4cd2-b6cd-09b8ee8aef5a

# Install 2: With only Poetry in the SNAP cluster

```bash
# - Install Lean4 manually (elan and lake), 1st one is the SNAP one, 2nd is the most common one
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
# curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s

# - Make sure Lean4 tools (lean, lake) are available 
export PATH="$HOME/.elan/bin:$PATH"
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
bash
elan
lake

# - Init the git submodules (i.e., make git aware of them/track them) + fetch/clone/update (and double check submodule is inited)
git submodule init
git submodule update --init

# - For snap make sure the repo is sym linked you're using your
git clone git@github.com:lenianiva/PyPantograph.git
git checkout <your-branch>
ln -s $AFS/PyPantograph $HOME/PyPantograph

# - Install poetry with python venv (needs seperate install so poetry & your projs deps don't crash)
mkdir $HOME/.virtualenvs

# - Put the follow BEFORE your conda init stuff in your .bashrc
export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry
export PATH="$VENV_PATH/bin:$PATH"

# - Now actually install poetry in a python env after creating an python env for poetry with venv
python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry

poetry

# poetry build is only needed when you build a python distribution e.g., .whl or .tar.gz and want to distribute it. You can't use those files for edtiable development anyway
# # - Build the PyPantograph proj (build the py distribution, py deps and custom (lean4) installs)
# cd $HOME/PyPantograph
# poetry build

# - Install pypantograph in editable mode with poetry
#Installs the project and its dependencies into the virtual environment, creating the environment if it doesn't exist, in editable mode. This will run our custom build for Lean already (the build.py file!)
poetry install 
# if it create a new python env, check it out
poetry env list
# activate the current poetry env in a new shell
poetry shell

# - Confirm intalls
# poetry show | grep pantograph # note, doesn't do anything since poetry already only works by installing things in editable mode
poetry show | grep vllm
poetry show | grep torch

# - Select freeiest GPU wrt vRAM
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
echo $CUDA_VISIBLE_DEVICES

# - Make sure the PyPantrograph server tests by Leni work
cd ~/PyPantograph
# python -m pantograph.server
python $HOME/PyPantograph/pantograph/server.py
python $HOME/PyPantograph/test_vllm.py
```