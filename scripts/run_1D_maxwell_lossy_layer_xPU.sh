#!/bin/bash -l
#SBATCH --job-name="1D_additive_source_lossy_layer"
#SBATCH --output=1D_additive_source_lossy_layer.%j.o
#SBATCH --error=1D_additive_source_lossy_layer.%j.e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda

srun julia -O3 1D_additive_source_lossy_layer.jl