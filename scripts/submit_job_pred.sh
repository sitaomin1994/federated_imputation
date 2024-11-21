#!/bin/bash
#SBATCH --job-name=cafe_np
#SBATCH --nodelist=hal0315
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sm2370@rutgers.edu
#SBATCH --partition=p_jsvaidya_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-00:00:00
#SBATCH --export=ALL

# Output and Error File Names
#SBATCH --output=./slurm/%N.%j.out  # STDOUT output file
#SBATCH --error=./slurm/%N.%j.err   # STDERR output file
dataset_name=$1
source /projects/community/miniconda3/bin/activate impute4fair
cd ~/Research/federated_imputation/

# Your command here
srun ./scripts/run_pred.sh "${dataset_name}"