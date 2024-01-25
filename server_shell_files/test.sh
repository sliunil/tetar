#!/bin/bash -l

##SBATCH --account 1 
#SBATCH --mail-type ALL 
#SBATCH --mail-user guillaume.guex@unil.ch

#SBATCH --chdir ./ 
#SBATCH --job-name segm_manifesto
#SBATCH --output 3.2_segm_manifesto.out

#SBATCH --partition cpu

#SBATCH --nodes 1 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G 
#SBATCH --time 24:00:00 
#SBATCH --export NONE

module load gcc/9.3.0 python/3.8.8

source ~/SemSim_AutoCor/env/bin/activate

python3 3.2_server.py
