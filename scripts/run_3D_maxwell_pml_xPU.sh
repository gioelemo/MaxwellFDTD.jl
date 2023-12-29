#!/bin/bash -l
#SBATCH --job-name="3D_maxwell_pml_xPU"
#SBATCH --output=3D_maxwell_pml_xPU.%j.o
#SBATCH --error=3D_maxwell_pml_xPU.%j.e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda

srun julia -O3 3D_maxwell_pml_xPU.jl