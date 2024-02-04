#!/bin/bash -l

##SBATCH --account 1
#SBATCH --mail-type ALL 
#SBATCH --mail-user guillaume.guex@unil.ch

#SBATCH --chdir ../src/
#SBATCH --job-name artificial_corpora_compute_mp
#SBATCH --output ../server_files/out_files/artificial_corpora_compute_mp_%A_%a.out

#SBATCH --partition cpu
#SBATCH --ntasks 1 

#SBATCH --cpus-per-task 32
#SBATCH --mem 32G 
#SBATCH --time 12:00:00 
#SBATCH --export NONE

#SBATCH --array=0-19

module load gcc python

source ~/tetar/.venv/bin/activate

python3 artificial_corpora_compute_res_server.py $SLURM_ARRAY_TASK_ID
