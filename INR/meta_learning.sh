#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-751 -p alvis
#SBATCH -t 10:00:00
#SBATCH -N 1 --gpus-per-node=A100:2
#SBATCH -J meta-learning

export LMOD_DISABLE_SAME_NAME_AUTOSWAP=no
# module load Python/3.10.4-GCCcore-11.3.0

echo 'Activate virtual environment'
source functa_venv/bin/activate
# pip install --no-cache-dir --no-build-isolation --upgrade --force-reinstall PyYAML
# pip install --no-cache-dir --no-build-isolation --upgrade --force-reinstall gast

echo 'df'
df
echo pip list
pip3 list
export PYTHONPATH=/cephyr/users/jingling/Alvis/INR
echo 'Run experiment test'
# run meta learning experiment
python3 -m functa.experiment_meta_learning --config=./functa/experiment_meta_learning.py
# deactivate virtual environment
deactivate
