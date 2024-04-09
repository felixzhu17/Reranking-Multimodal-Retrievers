
import json
from tqdm import tqdm

input_files = [
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_InfoseekDatasetForDPR.train_predictions_rank_0.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_InfoseekDatasetForDPR.train_predictions_rank_1.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_InfoseekDatasetForDPR.train_predictions_rank_2.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_InfoseekDatasetForDPR.train_predictions_rank_3.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_InfoseekDatasetForDPR.valid_predictions_rank_0.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_InfoseekDatasetForDPR.valid_predictions_rank_1.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_InfoseekDatasetForDPR.valid_predictions_rank_2.json",
    # "/data/project_data/experiments/AWS_Finetune_Infoseek_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_InfoseekDatasetForDPR.valid_predictions_rank_3.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_0.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_1.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_2.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_train_index/test/generate_train_index/generate_train_index_test_EVQADatasetForDPR.train_predictions_rank_3.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_0.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_1.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_2.json",
    # "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_valid_index/test/generate_valid_index/generate_valid_index_test_EVQADatasetForDPR.valid_predictions_rank_3.json",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_0.json",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_1.json",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_2.json",
    "/home/ubuntu/additional_data/experiments/AWS_Finetune_EVQA_ViT(G)_NewFLMR(mask_instruct)_ColBERT(v2)_pretraining_ViT_frozen_lr=1e-5_mapping_lr=1e-5_load(4ymcu4na)_test_index/test/generate_test_index/generate_test_index_test_EVQADatasetForDPR.valid_predictions_rank_3.json",
]

for file_path in input_files:
    print(file_path)
    target_file_path = file_path.replace(".json", ".pkl")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # data = data['output']
    new_data = {
        'output': [],
    }
    for prediction in tqdm(data['output']):
        # print(prediction.keys())
        for pred in prediction['top_ranking_passages']:
            pred.pop("content")
        new_data['output'].append(prediction)
        
    # save new data to target_file_path
    print(f"Saving to {target_file_path}")
    # save as pickle
    import pickle
    with open(target_file_path, 'wb') as f:
        pickle.dump(new_data, f)