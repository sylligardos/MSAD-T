#!/bin/bash
#SBATCH --job-name=regression          # Name of the job
#SBATCH --output=experiments/regression_16_12_2024/%j.out      # File to store output
#SBATCH --error=experiments/regression_16_12_2024/%j.out       # File to store errors
#SBATCH --ntasks=1                   # Number of tasks (1 for CPU)
#SBATCH --cpus-per-task=16           # Number of CPU cores per task (adjust as needed)
#SBATCH --time=20:00:00              # Time limit (2 hours in this case)
#SBATCH -A gpr@v100                   # Specify the account to use (CPU account)
#SBATCH --gres=gpu:1                  # Request 1 GPU

# go into the submission directory 
cd ${SLURM_SUBMIT_DIR}

# clean out the modules loaded in interactive and inherited by default
module purge

# loading the modules
source ~/Documents/miniconda3/etc/profile.d/conda.sh
conda activate MSAD-T

# echo of launched commands
set -x

# execution
python3 src/train_regressors.py -m 2 -c raw -d norma -w 128 -e supervised -p experiments/regression_16_12_2024/ -t True