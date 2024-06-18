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

    experiment_name = base_name
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

job_template = """
python src/main.py --config {config_file} --mode train --experiment_name DEBUGDUMMY --reset --override {opts_str}
"""

def main(config_name):
    config_file = os.path.join('job_configs', f"{config_name}.json")
    configs = load_configs(config_file)
    base_experiment_name = get_base_experiment_name(config_file)


    for config in configs:
        opts = config.get("opts", "")
        
        model_path = config.get("ckpt_path")
        experiment_name = generate_experiment_name(base_experiment_name, opts, model_path)
        if os.path.isdir(os.path.join('experiments', experiment_name)):
            raise ValueError(f"Experiment directory {experiment_name} already exists.")
        
        assert config["config_file"], "Config file not specified."
        assert os.path.exists(config["config_file"]), f"Config file {config['config_file']} does not exist."
        
        
        opts_str = "--opts train.trainer_paras.limit_val_batches=2 train.trainer_paras.limit_train_batches=5 train.trainer_paras.max_epochs=1"
        
        if opts:
            opts_str += f' {opts}'
        
        if model_path:
            assert os.path.exists(model_path), f"Checkpoint path {model_path} does not exist."
            model_opts = f'train.load_model_path="{model_path}"'
            opts_str += f' {model_opts}'
                
        job_script = job_template.format(
            config_file=config["config_file"],
            opts_str=opts_str,
        )
        print(job_script)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job scripts for experiments.")
    parser.add_argument("--config_name", required=True, help="Name of the config file without extension")
    args = parser.parse_args()
    main(args.config_name)
