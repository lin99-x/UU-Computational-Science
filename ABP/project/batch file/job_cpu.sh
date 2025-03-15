#!/bin/bash -l
#SBATCH -A uppmax2023-2-36  # project name
#SBATCH -M snowy            # name of system
#SBATCH -p node             # request a full node
#SBATCH -N 1                # request 1 node
#SBATCH -t 1:00:00          # job takes at most 1 hour
#SBATCH -J project_host      # name of the job
#SBATCH -D ./               # stay in current working directory

export OMP_PROC_BIND=spread and OMP_PLACES=threads
export OMP_NUM_THREADS=16
./app.host -N 24 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 48 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 72 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 96 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 120 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 144 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 168 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 192 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 216 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 240 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 264 -repeat 100 -number float >> fem_application_host_float.log
./app.host -N 288 -repeat 100 -number float >> fem_application_host_float.log