#!/bin/bash -l

#SBATCH -M snowy
#SBATCH -A uppmax2023-2-13
#SBATCH -p core -n 2
#SBATCH -t 15:00

module load gcc openmpi
mpirun ./matmul_non /proj/uppmax2023-2-13/nobackup/matmul_indata/input3600.txt out.txt