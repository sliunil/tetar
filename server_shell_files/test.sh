#!/bin/bash -l

##SBATCH --account 1 
#SBATCH --mail-type ALL 
#SBATCH --mail-user guillaume.guex@unil.ch

#SBATCH --chdir ../src/
#SBATCH --job-name sample_size_compute
#SBATCH --output sample_size_compute.out

#SBATCH --partition cpu

#SBATCH --nodes 1 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G 
#SBATCH --time 24:00:00 
#SBATCH --export NONE

module load gcc python

source ~/tetar/.env/bin/activate

python3 sample_size_compute_res.py
