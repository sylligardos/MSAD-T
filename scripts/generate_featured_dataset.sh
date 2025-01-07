#!/bin/bash
#SBATCH --job-name=features          # Name of the job
#SBATCH --output=features%j.out      # File to store output
#SBATCH --error=features%j.out       # File to store errors
#SBATCH --ntasks=1                   # Number of tasks (1 for CPU)
#SBATCH --cpus-per-task=32           # Number of CPU cores per task (adjust as needed)
#SBATCH --time=20:00:00              # Time limit (2 hours in this case)
#SBATCH -A gpr@cpu                   # Specify the account to use (CPU account)

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
python3 src/generate_featured_dataset.py -p "data/TSB_16/" -s "data/features/" -f catch22
