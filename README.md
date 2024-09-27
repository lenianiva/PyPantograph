# PyPantograph

Python interface to the Pantograph library

# Getting started
Note: For [self-contained `install.sh` script for lean](https://github.com/brando90/learning_lean/blob/main/install.sh).

## Install 1: With Conda and Pip in the SNAP cluster

```bash
# - Install Lean 4 manually (elan & lake): gets install script (doesn't save it) & directly gives it to sh to install it 
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
# (install command from the Lean4 official instlal guide, not the one we use)
# curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s

# - Make sure Lean4 tools (lean, lake) are available 
echo $PATH | tr ':' '\n'
export PATH="$HOME/.elan/bin:$PATH"
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
# bash
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
# git checkout <your-branch>
git checkout brando
ln -s $AFS/PyPantograph $HOME/PyPantograph

# - Build the PyPantograph proj (build the py distribution, py deps and custom (lean4) installs). Note: pip install -e doesn't work on the dist .whl builds etc so you instead the next command
cd $HOME/PyPantograph
poetry build

```
To run server tests:
``` bash
python -m pantograph.server
python -m pantograph.search
```
The tests in `pantograph/server.py` also serve as simple interaction examples

## Examples

For API interaction examples, see `examples/README.md`

An agent based on the `sglang` library is provided in
`pantograph/search_llm.py`. To use this agent, set the environment variable
`OPENAI_API_KEY`, and run
```bash
python3 -m pantograph.search_llm
```

## Experiments

In `experiments/`, there is an experiment on running a LLM prover on miniF2F
data. Run with

```sh
python3 experiments/miniF2F_search.py [--dry-run]
```

## Referencing

```bib
@misc{pantograph,
	title = "Pantograph, A Machine-to-Machine Interface for Lean 4",
	author = {Aniva, Leni and Miranda, Brando and Sun, Chuyue},
	year = 2024,
	howpublished = {\url{https://github.com/lenianiva/PyPantograph}}
}
```

# - Install pypantograph in editable mode (only pyproject.toml (or setup.py!) needed! Assuming your at the proj root)
cd $HOME/PyPantograph
pip install -e . 

# - Confirm intalls
pip list | grep pantograph
pip list | grep vllm
pip list | grep torch

# - Make sure the PyPantrograph server tests by Leni work
cd ~/PyPantograph
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
git submodule update --init --recursive
git clone git@github.com:lenianiva/PyPantograph.git --recurse-submodules

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
cd $HOME/PyPantograph
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

# - Make sure the PyPantrograph server tests by Leni work
cd ~/PyPantograph
python -m pantograph.server
# python $HOME/PyPantograph/pantograph/server.py
# python $HOME/PyPantograph/test_vllm.py
