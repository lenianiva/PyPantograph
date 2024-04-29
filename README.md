# PyPantograph

Python interface to the Pantograph library

## Getting started

First initialize the git submodules so that git can keep track of the submodules being used do:
```bash
# - initialize the git submodules by preparing the git repository, but it does not clone or fetch them, just init's git's internal configs
git submodule init
```
Then to clone, fetch & update the submodules code (and also initilize anything you might have forgotten that is specificed in the `.gitmodules` file):
```bash
# - initialize the git submodules so that git can track them and then the update clone/fetches & updates the submodules
git submodule update --init
```

Then install poetry by [following instructions written for the Stanford SNAP cluster](https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#poetry) or [their official instructions](https://python-poetry.org/docs/#installing-manually). 
Then once you confirm you have poetry & the initialized git submodules, execute:
```bash
poetry build
```
To run server tests:
``` bash
python -m pantograph.server
```
The tests in `pantograph/server.py` also serve as simple interaction examples


## Install 2: With Conda and Pip in the SNAP cluster

```bash
# install Lean4 manually (elan and lake)
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y

# create and activate the right python env (this is needed so that poetry build works)
conda create -n pypantograph_env python=3.11 -y
conda activate pypantograph_env

# build the PyPantograph proj (build the py distribution, py deps and custom (lean4) installs)
cd $HOME/PyPantograph
poetry build

# install pypantograph in editable mode (only pyproject.toml needed! Assuming your at the proj root)
pip install -e . 

# confirm intalls
pip list | grep pantograph
pip list | grep vllm
pip list | greo torch

# make sure the PyPantrograph server tests by Leni work
python -m server.py
```
