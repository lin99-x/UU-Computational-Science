#!/bin/bash -l
#SBATCH -A uppmax2023-2-36 # project name 
#SBATCH -M snowy # name of system 
#SBATCH -p node # request a full node 
#SBATCH -N 1 # request 1 node 
#SBATCH -t 1:00:00 # job takes at most 1 hour 
#SBATCH --gres=gpu:1 --gpus-per-node=1 # use the GPU nodes
#SBATCH -J project_cuda # name of the job 
#SBATCH -D ./ # stay in current working directory 

./app.cuda -N 32 -repeat 100 -number double >> fem_application_cuda_sellc_double.log
./app.cuda -N 64 -repeat 100 -number double >> fem_application_cuda_sellc_double.log
./app.cuda -N 96 -repeat 100 -number double >> fem_application_cuda_sellc_double.log
./app.cuda -N 128 -repeat 100 -number double >> fem_application_cuda_sellc_double.log
./app.cuda -N 192 -repeat 100 -number double >> fem_application_cuda_sellc_double.log
./app.cuda -N 256 -repeat 100 -number double >> fem_application_cuda_sellc_double.log
