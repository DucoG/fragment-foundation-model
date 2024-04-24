#!/bin/bash

# SLURM SUBMIT SCRIPT

#SBATCH --nodes=1
#SBATCH --partition=oncode
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=sbatch_out/pretrain_fragformer/pretrain_fragformer_%A.out
#SBATCH --error=sbatch_out/pretrain_fragformer/pretrain_fragformer_%A.err

# make sure that we are in the correct directory
cd /home/d.gaillard/projects/fragment_autoencoder/fragment_foundation_model

HYDRA_FULL_ERROR=1
EXPERIMENT_NAME="pretrain_fragformer"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H-%M-%S)

echo 'Start script:'
echo 'current time' $(date)

HYDRA_FULL_ERROR=1 
srun python pretrain.py task_name=$EXPERIMENT_NAME

echo 'current time' $(date)
echo 'Finished'
