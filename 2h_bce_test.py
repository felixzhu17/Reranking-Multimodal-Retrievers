# /home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMRQuery_Full_Context_Rerank_2H_BCE_ckpt_model_step_1000/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json
# /home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMRQuery_Full_Context_Rerank_ckpt_model_step_1007/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json
import json
from collections import defaultdict

RERANKER_RESULTS = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMRQuery_Full_Context_Retrieved_Rerank_ckpt_model_step_1000/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"
CORRECT_PASSAGE_IDS = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Interaction_MORES_5_B_ckpt_model_step_3021/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"

# Load both JSON files
with open(RERANKER_RESULTS, "rb") as f:
    data = json.load(f)['output']

with open(CORRECT_PASSAGE_IDS, "rb") as f:
    correct_data = json.load(f)['output']

# Create a mapping of question_id to correct pos_item_ids
correct_pos_item_ids_map = {entry['question_id']: entry['pos_item_ids'] for entry in correct_data}

def calculate_correctness(passage_list, pos_item_ids):
    return min(1, sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids))

def calculate_recall(grouped_data):
    correct_count = len(grouped_data[1])
    incorrect_count = len(grouped_data[0])
    recall = correct_count / (incorrect_count + correct_count)
    return recall

def evaluate_recall_at_k(k):
    grouped_data = defaultdict(list)
    for entry in data:
        top_k_top_ranking = entry['top_ranking_passages'][:k]
        correct_pos_item_ids = correct_pos_item_ids_map.get(entry['question_id'], [])
        correct_top_ranking = calculate_correctness(top_k_top_ranking, correct_pos_item_ids)
        grouped_data[correct_top_ranking].append(entry)

    recall = calculate_recall(grouped_data)
    return recall, grouped_data

# Print recall@k
for k in [5, 10, 20, 50, 100]:
    recall, grouped_data = evaluate_recall_at_k(k)
    print(f"Recall@{k}: {recall}")