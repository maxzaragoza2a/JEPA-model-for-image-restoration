#!/bin/bash
# Pre-training job for a SLURM cluster, one GPU.
#
#   sbatch scripts/train_slurm.sh                      # RUN below
#   sbatch --export=ALL,RUN=stl10_d256 scripts/train_slurm.sh
#   sbatch --export=ALL,MAX_STEPS=50 scripts/train_slurm.sh   # smoke test
#
# The full run passes --resume: when it hits the time limit, submit the
# same command again and training goes on from the last finished epoch.
# The log is appended to, and plot_loss.py keeps the last value of each step.
#
# To adapt to the cluster: the partition, the time limit, and how Python is
# set up (module load / conda activate). Submit from the project root.
#
# STL-10 must already be in $TFDS_DATA_DIR: compute nodes often have no
# internet. Download it once from the login node:
#   TFDS_DATA_DIR=$HOME/tensorflow_datasets python -c \
#     "import tensorflow_datasets as tfds; tfds.load('stl10', split='unlabelled')"

#SBATCH --job-name=ijepa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%j.out
##SBATCH --partition=gpu

set -euo pipefail

RUN=${RUN:-stl10_vits}
export TFDS_DATA_DIR=${TFDS_DATA_DIR:-$HOME/tensorflow_datasets}

# source ~/miniconda3/etc/profile.d/conda.sh && conda activate ijepa

mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv

ARGS=(--run "$RUN")
if [[ -n "${MAX_STEPS:-}" ]]; then
    # Smoke test: its own folder, never resumed, so it never mixes with the
    # real run.
    ARGS+=(--max-steps "$MAX_STEPS" --out "checkpoints/${RUN}_smoke")
    LOG="logs/${RUN}_smoke.log"
else
    ARGS+=(--resume)
    LOG="logs/${RUN}.log"
fi

python -u -m jepa.train "${ARGS[@]}" >> "$LOG" 2>&1
