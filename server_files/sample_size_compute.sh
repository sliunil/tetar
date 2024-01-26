#!/bin/bash -l

##SBATCH --account 1 
#SBATCH --mail-type ALL 
#SBATCH --mail-user guillaume.guex@unil.ch

#SBATCH --chdir ../src/
#SBATCH --job-name sample_size_compute
#SBATCH --output ../server_files/sample_size_compute.out

#SBATCH --partition cpu

#SBATCH --nodes 1 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G 
#SBATCH --time 48:00:00 
#SBATCH --export NONE

module load gcc python

source ~/tetar/.venv/bin/activate

python3 sample_size_compute_res.py
