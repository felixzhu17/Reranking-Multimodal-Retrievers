import os
import json
import argparse

def load_configs(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def generate_experiment_name(base_name, opts, model_path=None):
    opts_parts = opts.split()
    opts_dict = {}
    for part in opts_parts:
        key_value = part.split('=')
        if len(key_value) == 2:
            key, value = key_value
            key = key.split('.')[-1]  # Use the last part of the key
            opts_dict[key] = value

    experiment_name = "TEST_" + base_name
    for key, value in opts_dict.items():
        experiment_name += f"_{key}{value.replace('.', '').replace('-', '')}"
    
    if model_path:
        model_step = model_path.split('/')[-1].split('.')[0]
        experiment_name += f"_ckpt_{model_step}"
    
    return experiment_name

def get_base_experiment_name(json_path):
    # Extract the base experiment name from the JSON file name
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    return base_name

job_template = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -A MLMI-fz288-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
source /home/${{USER}}/.bashrc

workdir="/home/fz288/rds/hpc-work/PreFLMR"
cd $workdir
echo -e "Changed directory to `pwd`.\n"

set --
conda deactivate
conda deactivate
source hpc/conda.sh

export OMP_NUM_THREADS=1

num_gpus=$SLURM_GPUS_ON_NODE
gpu_indices=$(seq -s, 0 $(($num_gpus - 1)))
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo -e "Timestamp: $TIMESTAMP\n======"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
python src/main.py --config {config_file} --mode test --experiment_name {experiment_name} --tags {tags} {opts_str} > log_{experiment_name} 2>&1
"""

def main(config_name):
    config_file = os.path.join('job_configs', f"{config_name}.json")
    configs = load_configs(config_file)
    base_experiment_name = get_base_experiment_name(config_file)
    tags = [base_experiment_name, 'test']
    tags_str = " ".join(f'"{tag}"' for tag in tags)  

    for config in configs:
        opts = config.get("opts", "")
        model_path = config.get("ckpt_path")
        experiment_name = generate_experiment_name(base_experiment_name, opts, model_path)
        if os.path.isdir(os.path.join('experiments', experiment_name)):
            raise ValueError("Experiment directory already exists.")
    
        assert config["config_file"], "Config file not specified."
        assert os.path.exists(config["config_file"]), f"Config file {config['config_file']} does not exist."
        
        if model_path:
            assert os.path.exists(model_path), f"Checkpoint path {model_path} does not exist."
            model_opts = f'train.load_model_path="{model_path}"'
            if opts:
                opts += f' {model_opts}'
            else:
                opts = model_opts
        
        opts_str = f"--opts {opts}" if opts else ""
        
        job_script = job_template.format(
            job_name=experiment_name,
            config_file=config["config_file"],
            experiment_name=experiment_name,
            opts_str=opts_str,
            time=config["time"],
            gpus=config["gpus"],
            tags=tags_str
        )
        
        script_name = os.path.join('job_scripts', config_name, f"{experiment_name}.sh")
        os.makedirs(os.path.dirname(script_name), exist_ok=True)
        with open(script_name, "w") as f:
            f.write(job_script)
        print(f"Generated job script: {script_name}")

        # Submit the job script
        os.system(f"sbatch {script_name}")
        print(f"Submitted job script: {script_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job scripts for experiments.")
    parser.add_argument("--config_name", required=True, help="Name of the config file without extension")
    args = parser.parse_args()
    main(args.config_name)
