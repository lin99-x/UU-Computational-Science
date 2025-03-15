#!/bin/bash -l

#SBATCH -M snowy
#SBATCH -A uppmax2023-2-13
#SBATCH -p core -n 32
#SBATCH -t 15:00

module load gcc openmpi
mpirun ./simulator 10000 test.txt