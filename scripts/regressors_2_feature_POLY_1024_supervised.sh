#!/bin/bash
#SBATCH --job-name=regressors_2_feature_POLY_1024_supervised               # Job name
#SBATCH --output=logs/%x.log        # Standard output and error log
#SBATCH --error=logs/%x.log         # Error log
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --time=20:00:00             # Time limit (2 hours in this case)
#SBATCH -A gpr@v100                 # Specify the account to use (CPU account)
#SBATCH --gres=gpu:1                # Request 1 GPU

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
python3 src/train_regressors.py --model_index 2 --model_class feature --detector POLY --window_size 1024 --experiment supervised --saving_path experiments/regression_16_12_2024