sh_templates = {
"jz_cpu": 
"""#!/bin/bash
#SBATCH --job-name={}               # Job name
#SBATCH --output={}/logs/%x.log        # Standard output and error log
#SBATCH --error={}/logs/%x.log         # Error log
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --time=20:00:00             # Time limit
#SBATCH -A gpr@cpu                  # Specify the account to use

# go into the submission directory 
cd ${{SLURM_SUBMIT_DIR}}

# clean out the modules loaded in interactive and inherited by default
module purge

# loading the modules
source ~/Documents/miniconda3/etc/profile.d/conda.sh
conda activate {}

# echo of launched commands
set -x

# execution
python3 {}""",

"jz_gpu": 
"""#!/bin/bash
#SBATCH --job-name={}               # Job name
#SBATCH --output={}/logs/%x.log        # Standard output and error log
#SBATCH --error={}/logs/%x.log         # Error log
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --time=20:00:00             # Time limit (2 hours in this case)
#SBATCH -A gpr@v100                 # Specify the account to use (CPU account)
#SBATCH --gres=gpu:1                # Request 1 GPU

# go into the submission directory 
cd ${{SLURM_SUBMIT_DIR}}

# clean out the modules loaded in interactive and inherited by default
module purge

# loading the modules
source ~/Documents/miniconda3/etc/profile.d/conda.sh
conda activate {}

# echo of launched commands
set -x

# execution
python3 {}""",

"cleps_cpu": 
"""#!/bin/bash
#SBATCH --job-name={}               # Job name
#SBATCH --output={}/logs/%x.log        # Standard output and error log
#SBATCH --error={}/logs/%x.log         # Error log
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=16        # Number of CPU cores per task
#SBATCH --mem=8G                # Memory per node
#SBATCH --time=2:00:00          # Time limit hrs:min:sec

# Activate the conda environment
source activate {}

# Run the Python script
python3 {}""",

    "cleps_gpu":
"""#!/bin/bash
#SBATCH --job-name={}               # Job name
#SBATCH --output={}/logs/%x.log        # Standard output and error log
#SBATCH --error={}/logs/%x.log         # Error log
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH --mem=32G                # Memory per node
#SBATCH --partition=gpu          # Partition name (gpu for GPU jobs)
#SBATCH --gres=gpu:1             # Number of GPUs (1 in this case)
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=48:00:00          # Time limit hrs:min:sec

# Activate the conda environment
source activate {}

# Run the Python script
python3 {}""" 
}