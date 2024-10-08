#!/bin/bash

# SLURM SUBMIT SCRIPT

#SBATCH --nodes=1
#SBATCH --partition=oncode
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=../sbatch_out/debug/debug_fragformer_%A.out
#SBATCH --error=../sbatch_out/debug/debug_fragformer_%A.err


EXPERIMENT_NAME="debug_fragformer"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H-%M-%S)

echo 'Start script:'
echo 'current time' $(date)

HYDRA_FULL_ERROR=1 
srun python pretrain.py task_name=$EXPERIMENT_NAME debug=default

echo 'current time' $(date)
echo 'Finished'
