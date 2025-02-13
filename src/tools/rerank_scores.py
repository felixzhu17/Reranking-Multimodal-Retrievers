import json
from collections import defaultdict

# Paths for different models
models = {
    "OKVQA": {
        "retrieval": [
            "/home/fz288/rds/hpc-work/PreFLMR/search_index/OKVQA/PreFLMR-B/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"
        ],
        "rerank": [
            "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMRQuery_Full_Context_Rerank_B_Neg_Sample_Train_on_Retrieve_ckpt_model_step_2014/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json",
            "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Decoder_Head_Rerank_Neg_Sample_Retrieved_ckpt_model_step_2000/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json",
            "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Interaction_MORES_5_B_Neg_Sample_Train_on_Retrieve_ckpt_model_step_6042/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"
        ]
    },
    "EVQA": {
        "retrieval": [
            "/home/fz288/rds/hpc-work/PreFLMR/search_index/EVQA/PreFLMR-B/_test_EVQADatasetForDPR.test_predictions_rank_0.json"
        ],
        "rerank": [
            "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_EVQA_FLMRQuery_Full_Context_Rerank_B_Freeze_Vision_ckpt_model_step_15000/test/_test_EVQADatasetForDPR.test_predictions_rank_0.json",
            "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_EVQA_Encoder_Decoder_Head_Rerank_Vanilla_Neg_Sample_Retrieved_ckpt_model_step_2000/test/_test_EVQADatasetForDPR.test_predictions_rank_0.json",
            "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_EVQA_Interaction_3_B_Vanilla_Neg_Sample_Retrieved_ckpt_model_step_4001/test/_test_EVQADatasetForDPR.test_predictions_rank_0.json"
        ]
    }
}

# Define a function to find the top D elements by scores
def get_top_d_elements(data, d):
    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
    return sorted_data[:d]

# Define a function to rerank the top D elements using the rerank results
def rerank_elements(question_id, top_d_elements, rerank_results_map):
    rerank_result = rerank_results_map[question_id]
    reranked_elements = []

    for item in top_d_elements:
        passage_id = item['passage_id']
        # Find the passage in the rerank results
        passage_in_rerank = next((p for p in rerank_result['top_ranking_passages'] if p['passage_id'] == passage_id), None)
        if passage_in_rerank is None:
            raise ValueError(f"Passage ID {passage_id} for question ID {question_id} not found in rerank results")
        
        reranked_elements.append({
            'passage_id': passage_id,
            'score': passage_in_rerank['score'],
            'pos_item_ids': rerank_result['pos_item_ids']
        })
    
    # Sort the reranked elements by the new score
    reranked_elements = sorted(reranked_elements, key=lambda x: x['score'], reverse=True)
    return reranked_elements

# Function to calculate correctness based on passage_id
def calculate_correctness(passage_list, pos_item_ids):
    return min(1, sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids))

# Function to calculate recall
def calculate_recall(grouped_data):
    correct_count = len(grouped_data[1])
    incorrect_count = len(grouped_data[0])
    recall = correct_count / (incorrect_count + correct_count)
    return correct_count, incorrect_count, recall

# Main function to perform the recall calculation for all models and d_values
def main(models, d_values):
    for dataset, paths in models.items():
        for retrieval_path in paths['retrieval']:
            for rerank_path in paths['rerank']:
                # Load the JSON data from the input files
                with open(retrieval_path, "r") as f:
                    retrieval_results = json.load(f)['output']
                
                with open(rerank_path, "r") as f:
                    rerank_results = json.load(f)['output']
                
                # Create a map for quick lookup of rerank results by question_id
                rerank_results_map = {entry['question_id']: entry for entry in rerank_results}
                
                print(f"\nDataset: {dataset}")
                print(f"Retrieval Path: {retrieval_path}")
                print(f"Rerank Path: {rerank_path}")
                
                for d in d_values:
                    grouped_data = defaultdict(list)
                    
                    for entry in retrieval_results:
                        question_id = entry['question_id']
                        if question_id not in rerank_results_map:
                            continue  # Skip if question_id is not in rerank results

                        top_d_elements = get_top_d_elements(entry['top_ranking_passages'], d)
                        reranked_elements = rerank_elements(question_id, top_d_elements, rerank_results_map)
                        
                        # Calculate correctness for top 5 elements
                        correct_top_ranking = calculate_correctness(reranked_elements[:5], reranked_elements[0]['pos_item_ids'])
                        grouped_data[correct_top_ranking].append(entry)
                    
                    # Calculate and print overall recall
                    correct_count, incorrect_count, recall = calculate_recall(grouped_data)
                    print(f"\nResults for D={d}:")
                    print(f"Recall: {recall}\n")

# Run the main function with the loaded data and desired D values
if __name__ == "__main__":
    d_values = [5, 10, 25, 50, 75, 100]  # Set the values of D here
    main(models, d_values)
