"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

import itertools
import os
import numpy as np
from templates import sh_templates

def main():
    current_dir = "scripts"
    saving_dir = "regressors"
    experiment_desc = {
        "job_name": "regressors",
        "environment": "MSAD-T",
        "script_name": "src/train_regressors.py",
        "args": {
            "model_index": [0, 1, 2],
            "detector": ['AE', 'CNN', 'HBOS', 'IFOREST', 'IFOREST1', 'LOF', 'LSTM', 'MP', 'NORMA', 'OCSVM', 'PCA', 'POLY'],
            "window_size": [16, 32, 64, 128, 256, 512, 768, 1024],
            "experiment": ['supervised'],
            "split": np.arange(0, 16),
            "saving_path": ["experiments/regression_08_01_2025"],
        },
        "gpu_required": "1 if \"model_class\" == \"raw\" else 1"
    }
    template = sh_templates['cleps_cpu']
    
        
    # Analyse json
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    args = experiment_desc["args"]
    args_saving_path = args['saving_path'][0]
    arg_names = list(args.keys())
    arg_values = list(args.values())
    gpu_required = experiment_desc["gpu_required"]

    # Generate all possible combinations of arguments
    combinations = list(itertools.product(*arg_values))
    
    # Create the commands
    jobs = set()
    for combination in combinations:
        cmd = f"{script_name}"
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
            rsh.write(template.format(job_name, args_saving_path, args_saving_path, environment, cmd))
        
        jobs.add(job_name)

    # Create sh file to conduct all experiments 
    run_all_sh = ""
    jobs = list(jobs)
    jobs.sort()
    for job in jobs:
        run_all_sh += f"sbatch {os.path.join(current_dir, saving_dir, f'{job}.sh')}\n"
    
    with open(os.path.join(saving_dir, f'conduct_{experiment_desc["job_name"]}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)
        

if __name__ == "__main__":
    main()