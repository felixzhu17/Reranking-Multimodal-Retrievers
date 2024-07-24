import pickle
import json
questionId2topPassages = {}
RERANKER_RESULTS = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Decoder_Head_Rerank_ckpt_model_step_2002/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"

with open(RERANKER_RESULTS, "rb") as f:
    data = json.load(f)['output']

# Function to calculate correctness based on passage_id
def calculate_correctness(passage_list, pos_item_ids):
    return sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids)

# Function to calculate pseudo_correctness based on passage content
def calculate_pseudo_correctness(passage_list, answer):
    return sum(1 for passage in passage_list if answer.lower() in passage['content'].lower())

# Process each entry
for entry in data:
    # Get the top 5 passages
    top_5_top_ranking = entry['top_ranking_passages'][:5]
    top_5_raw_top_ranking = entry['raw_top_ranking_passages'][:5]
    
    # Calculate correctness
    correct_top_ranking = calculate_correctness(top_5_top_ranking, entry['pos_item_ids'])
    correct_raw_top_ranking = calculate_correctness(top_5_raw_top_ranking, entry['pos_item_ids'])
    
    # Calculate pseudo_correctness
    pseudo_correct_top_ranking = calculate_pseudo_correctness(top_5_top_ranking, entry['answers'][0])
    pseudo_correct_raw_top_ranking = calculate_pseudo_correctness(top_5_raw_top_ranking, entry['answers'][0])
    
    # Calculate improvement
    improvement = correct_top_ranking - correct_raw_top_ranking
    
    # Calculate pseudo_improvement
    pseudo_improvement = pseudo_correct_top_ranking - pseudo_correct_raw_top_ranking
    
    # Add results to entry
    entry['correct_top_ranking'] = correct_top_ranking
    entry['correct_raw_top_ranking'] = correct_raw_top_ranking
    entry['improvement'] = improvement
    entry['pseudo_correct_top_ranking'] = pseudo_correct_top_ranking
    entry['pseudo_correct_raw_top_ranking'] = pseudo_correct_raw_top_ranking
    entry['pseudo_improvement'] = pseudo_improvement
    entry['improvement_diff'] = improvement - pseudo_improvement

# Sort data by the difference between improvement and pseudo_improvement
sorted_data = sorted(data, key=lambda x: x['improvement_diff'], reverse=True)

# Compute total and average improvements and pseudo-improvements
total_improvement = sum(entry['improvement'] for entry in sorted_data)
total_pseudo_improvement = sum(entry['pseudo_improvement'] for entry in sorted_data)
num_entries = len(sorted_data)

average_improvement = total_improvement / num_entries
average_pseudo_improvement = total_pseudo_improvement / num_entries

# Function to print details for a given index
def print_details_for_index(data, index):
    if index < 0 or index >= len(data):
        print(f"Index {index} is out of range.")
        return
    
    entry = data[index]
    
    question = entry['question']
    answers = entry['answers']
    gold_answer = entry['gold_answer']
    pos_item_ids = entry['pos_item_ids']
    top_ranking_passages = entry['top_ranking_passages'][:5]
    raw_top_ranking_passages = entry['raw_top_ranking_passages'][:5]
    
    print(f"Question ID: {entry['question_id']}")
    print(f"Question: {question}")
    print(f"Answers: {answers}")
    print(f"Gold Answer: {gold_answer}")
    print("\nTop 5 Reranked Passages:")
    for passage in top_ranking_passages:
        correct = "Correct" if passage['passage_id'] in pos_item_ids else "Incorrect"
        pseudo_correct = "Pseudo Correct" if any(answer.lower() in passage['content'].lower() for answer in answers) else "Pseudo Incorrect"
        print(f"  Status: {correct}, Pseudo Status: {pseudo_correct}, Passage ID: {passage['passage_id']}, Content: {passage['content']}")
    
    print("\nTop 5 Retrieval Passages:")
    for passage in raw_top_ranking_passages:
        correct = "Correct" if passage['passage_id'] in pos_item_ids else "Incorrect"
        pseudo_correct = "Pseudo Correct" if any(answer.lower() in passage['content'].lower() for answer in answers) else "Pseudo Incorrect"
        print(f"  Status: {correct}, Pseudo Status: {pseudo_correct}, Passage ID: {passage['passage_id']}, Content: {passage['content']}")
    
    print(f"\nCorrect Reranked Top: {entry['correct_top_ranking']}")
    print(f"Correct Retrieval Top: {entry['correct_raw_top_ranking']}")
    print(f"Improvement: {entry['improvement']}")
    print(f"Pseudo Correct Reranked Top: {entry['pseudo_correct_top_ranking']}")
    print(f"Pseudo Correct Retrieval Top: {entry['pseudo_correct_raw_top_ranking']}")
    print(f"Pseudo Improvement: {entry['pseudo_improvement']}")
    print(f"Improvement Difference: {entry['improvement_diff']}")

# Example usage: print details for the first entry in the sorted data
index = 2 # Change this to the desired index
print_details_for_index(sorted_data, index)

# Print the average improvements
print(f"Average Improvement: {average_improvement}")
print(f"Average Pseudo Improvement: {average_pseudo_improvement}")

# Determine whether there is more pseudo-improvement or normal improvement
if average_improvement > average_pseudo_improvement:
    print("There is more normal improvement in the dataset.")
else:
    print("There is more pseudo-improvement in the dataset.")