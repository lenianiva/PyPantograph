#!/bin/bash
# SNAP cluster specific main sh script to run jobs
# - snap: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers and support il-action@cs.stanford.edu
# - live server stats: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-gpu-servers-stats

# source $AFS/.bashrc.lfs
source $AFS/.bash_profile
conda activate pypantograph_env
export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

# -- Run
# - Pull merge with PyPantograph main
cd $HOME/PyPantograph
# git checkout brando
git pull origin main
git submodule update --init
conda activate pypantograph_env
poetry build
pip install -e $HOME/PyPantograph
pip install -e $HOME/gold-ai-olympiad
python $HOME/PyPantograph/pantograph/server.py

# -- Demo
# python ~/snap-cluster-setup/src/train/simple_train_train.py
# CUDA_VISIBLE_DEVICES=5 python src/test_vllm.py 