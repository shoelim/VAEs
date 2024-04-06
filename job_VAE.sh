#!/bin/env bash
#SBATCH -A NAISS2023-5-275         # find your project with the "projinfo" command
#SBATCH -p alvis                   # what partition to use (usually not needed)
#SBATCH -t 0-23:55:00              # how long time it will take to run
#SBATCH --gpus-per-node=V100:1     # choosing no. GPUs and their type
#SBATCH -J vae                     # the jobname (not needed)
#SBATCH --output="vae.out"

# Build our container first
#apptainer build ours.sif ours.def

# Run within the container
apptainer exec /cephyr/users/soonh/Alvis/VAE/ours.sif python main.py --save_path /mimer/NOBACKUP/groups/snic2021-7-147/VAE-train --dataset 'swissroll' --num_epochs 100 
#apptainer exec /cephyr/users/soonh/Alvis/VAE/ours.sif python main_mnist.py 
