import torch


check_point_to_refactor = "experiments/OKVQA_WITCorpus_ColBERT_with_text_based_vision_freeze_mapping/train/saved_models/model_step_5000.ckpt"
output_path = check_point_to_refactor + '.refactored'

ckpt = torch.load(check_point_to_refactor, map_location='cpu')
state_dict = ckpt['state_dict']
new_state_dict = {}
for name, param in state_dict.items():
    if 'model.vision_projection' in name:
        rename = name.replace("model.vision_projection", "model.doc_vision_projection")
        # copy the params
        new_state_dict[rename] = state_dict[name]
        print(f"Copying {name} to {rename}")
    
    # copy the params
    new_state_dict[name] = state_dict[name]
        
ckpt['state_dict'] = new_state_dict
torch.save(ckpt, output_path)
