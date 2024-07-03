import json

# File paths
RERANKER_RESULTS = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_EVQA_FLMRQuery_Full_Context_Rerank_B_Freeze_Vision_ckpt_model_step_2000/test/_test_EVQADatasetForDPR.test_predictions_rank_0.json"
RERANKER_2_RESULTS = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_EVQA_Interaction_3_B_ckpt_model_step_13003/test/_test_EVQADatasetForDPR.test_predictions_rank_0.json"

# Load the data from both files
with open(RERANKER_RESULTS, "r") as f:
    data1 = json.load(f)['output']

with open(RERANKER_2_RESULTS, "r") as f:
    data2 = json.load(f)['output']

# Extract question_ids from both datasets
question_ids1 = {item['question_id'] for item in data1}
question_ids2 = {item['question_id'] for item in data2}

# Find differences and overlap
only_in_file1 = question_ids1 - question_ids2
only_in_file2 = question_ids2 - question_ids1
overlap = question_ids1 & question_ids2

# Print results
print(f"Number of question_ids only in file 1: {len(only_in_file1)}")
print(f"Number of question_ids only in file 2: {len(only_in_file2)}")
print(f"Number of overlapping question_ids: {len(overlap)}")
