import torch

# Specify the path to your checkpoint file
checkpoint_path = 'experiments/Oven_PreFLMR/train/saved_models/model_step_750.ckpt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Print out the keys in the checkpoint to see what it contains
print("Keys in the checkpoint:", checkpoint.keys())

# # To get a detailed look at specific contents, you can print them out
# # For example, to check the state_dict:
# if 'state_dict' in checkpoint:
#     print("Model state_dict keys:", checkpoint['state_dict'].keys())

# If you want to check the optimizer state:
if 'optimizer_states' in checkpoint:
    print("Optimizer state keys:", len(checkpoint['optimizer_states']))

# To check hyperparameters if they are saved in the checkpoint
if 'hyper_parameters' in checkpoint:
    print("Hyperparameters:", checkpoint['hyper_parameters'])
