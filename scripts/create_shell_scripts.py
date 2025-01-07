"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

import itertools
import os
import numpy as np


def main():
    saving_dir = "regressors"
    experiment_desc = {
        "job_name": "regressors",
        "environment": "MSAD-T",
        "script_name": "src/train_regressors.py",
        "args": {
            "model_index": [0, 1, 2],
            "model_class": ["raw", "feature"],
            "detector": ['AE', 'CNN', 'HBOS', 'IFOREST', 'IFOREST1', 'LOF', 'LSTM', 'MP', 'NORMA', 'OCSVM', 'PCA', 'POLY'],
            "window_size": [16, 32, 64, 128, 256, 512, 768, 1024],
            "experiment": ['supervised'], #, 'unsupervised'
            # "split": np.arange(0, 16),
            "saving_path": ["experiments/regression_16_12_2024"],
        },
        "gpu_required": "1 if \"model_class\" == \"raw\" else 1"
    }
    sh_file_templates = [
"""#!/bin/bash
#SBATCH --job-name={}               # Job name
#SBATCH --output=logs/%x.log        # Standard output and error log
#SBATCH --error=logs/%x.log         # Error log
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --time=20:00:00             # Time limit (2 hours in this case)
#SBATCH -A gpr@cpu                  # Specify the account to use (CPU account)

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
"""#!/bin/bash
#SBATCH --job-name={}               # Job name
#SBATCH --output=logs/%x.log        # Standard output and error log
#SBATCH --error=logs/%x.log         # Error log
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
python3 {}"""
    ]
    
    # Analyse json
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    args = experiment_desc["args"]
    arg_names = list(args.keys())
    arg_values = list(args.values())
    gpu_required = experiment_desc["gpu_required"]

    # Generate all possible combinations of arguments
    combinations = list(itertools.product(*arg_values))
    
    # Create the commands
    jobs = []
    for combination in combinations:
        cmd = f"{script_name}"
        gpu_required = experiment_desc["gpu_required"]
        job_name = experiment_desc["job_name"]

        supervised_flag = 0
        for name, value in zip(arg_names, combination):
            if name == 'experiment' and value == 'supervised':
                supervised_flag = 1
            if supervised_flag and name == 'split':
                continue
            cmd += f" --{name} {value}"
            if name == 'saving_path':
                continue

            job_name += f"_{value}"

            if isinstance(gpu_required, str) and name in gpu_required:
                gpu_required = int(eval(gpu_required.replace(name, str(value))))

        # Create saving dir if doesn't exist
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
            
        # Write the .sh file
        with open(os.path.join(saving_dir, f'{job_name}.sh'), 'w') as rsh:
            rsh.write(sh_file_templates[gpu_required].format(job_name, environment, cmd))
        
        jobs.append(job_name)

    # Create sh file to conduct all experiments 
    run_all_sh = ""
    for job in jobs:
        run_all_sh += f"sbatch {os.path.join(saving_dir, f'{job}.sh')}\n"
    
    with open(os.path.join(saving_dir, f'conduct_{experiment_desc["job_name"]}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)
        

if __name__ == "__main__":
    main()