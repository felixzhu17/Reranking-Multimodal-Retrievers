import json
import math
from collections import defaultdict
import numpy as np

RERANKER_RESULTS = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMRQuery_Full_Context_Rerank_ckpt_model_step_8048/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"

with open(RERANKER_RESULTS, "rb") as f:
    data = json.load(f)['output']

# Function to calculate correctness based on passage_id
def calculate_correctness(passage_list, pos_item_ids):
    return min(1, sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids))

# Function to calculate pseudo_correctness based on passage content
def calculate_pseudo_correctness(passage_list, answer):
    return min(1, sum(1 for passage in passage_list if answer.lower() in passage['content'].lower()))

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

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Modify passages and calculate loss
def modify_passages_and_calculate_loss(data):
    for item in data:
        pos_item_ids = set(item['pos_item_ids'])
        
        for key in ['top_ranking_passages']:
            passages = item[key]
            for passage in passages:
                passage_id = passage['passage_id']
                raw_score = passage['score']
                
                # Apply sigmoid function to the raw score
                score = sigmoid(raw_score)
             
                # Determine if passage is positive
                label = 1 if passage_id in pos_item_ids else 0
                
                # Calculate binary cross entropy loss
                loss = -math.log(score) if label == 1 else -math.log(1 - score)
                
                # Add label and loss to the dictionary
                passage['label'] = label
                passage['loss'] = loss
    
    return data

# Calculate average loss per item
def calculate_average_loss_per_item(data):
    for item in data:
        for key in ['top_ranking_passages']:
            passages = item[key]
            total_loss = sum(passage['loss'] for passage in passages)
            average_loss = total_loss / len(passages)
            item[f'{key}_average_loss'] = average_loss
    
    return data

# Calculate overall average loss
def calculate_overall_average_loss(data):
    total_top_loss = 0
    total_top_count = 0

    for item in data:
        total_top_loss += item['top_ranking_passages_average_loss'] * len(item['top_ranking_passages'])
        total_top_count += len(item['top_ranking_passages'])

    overall_top_average_loss = total_top_loss / total_top_count if total_top_count > 0 else 0
    return overall_top_average_loss

# Find examples with high and low loss
def find_examples_with_high_low_loss(data, loss_key='top_ranking_passages_average_loss', correct_key='correct_top_ranking', threshold=1.0, cutoff=10):
    high_correct = [item for item in data if item[correct_key] >= threshold]
    low_correct = [item for item in data if item[correct_key] == 0]
    
    sorted_high_correct_by_loss = sorted(high_correct, key=lambda x: x[loss_key], reverse=True)[:cutoff]
    sorted_low_correct_by_loss = sorted(low_correct, key=lambda x: x[loss_key])[:cutoff]
    
    return sorted_high_correct_by_loss, sorted_low_correct_by_loss

# Function to group by correct top ranking and calculate average and median loss
def group_by_correct_top_ranking_and_calculate_average_loss(data):
    grouped_data = defaultdict(list)
    
    for item in data:
        correct_top_ranking = item['correct_top_ranking']
        average_loss = item['top_ranking_passages_average_loss']
        grouped_data[correct_top_ranking].append(average_loss)
    
    average_losses_by_correct_top_ranking = {}
    median_losses_by_correct_top_ranking = {}
    count_by_correct_top_ranking = {}
    
    for correct_top_ranking, losses in grouped_data.items():
        average_losses_by_correct_top_ranking[correct_top_ranking] = np.mean(losses)
        median_losses_by_correct_top_ranking[correct_top_ranking] = np.median(losses)
        count_by_correct_top_ranking[correct_top_ranking] = len(losses)
    
    return average_losses_by_correct_top_ranking, median_losses_by_correct_top_ranking, count_by_correct_top_ranking


# Step 1: Modify passages and calculate loss
modified_data = modify_passages_and_calculate_loss(data)

# Step 2: Calculate average loss per item
data_with_avg_loss = calculate_average_loss_per_item(modified_data)

# Step 3: Calculate overall average loss across all items
overall_top_avg_loss = calculate_overall_average_loss(data_with_avg_loss)

# Print the overall average losses
print("Overall average loss for top_ranking_passages:", overall_top_avg_loss)


# Step 4: Group by correct top ranking and calculate average and median loss
average_losses_by_correct_top_ranking, median_losses_by_correct_top_ranking, count_by_correct_top_ranking = group_by_correct_top_ranking_and_calculate_average_loss(data_with_avg_loss)

# Print the average and median losses by correct top ranking
print("\nAverage and median loss by correct_top_ranking:")
for correct_top_ranking in sorted(average_losses_by_correct_top_ranking.keys()):
    avg_loss = average_losses_by_correct_top_ranking[correct_top_ranking]
    median_loss = median_losses_by_correct_top_ranking[correct_top_ranking]
    count = count_by_correct_top_ranking[correct_top_ranking]
    print(f"Correct Top Ranking: {correct_top_ranking}, Average Loss: {avg_loss}, Median Loss: {median_loss}, Count: {count}")
    
# Example usage of finding examples with high and low loss
high_loss_high_correct, low_loss_low_correct = find_examples_with_high_low_loss(data, threshold=1, cutoff=2)

print("\nHigh loss and high correct_top_ranking examples:")
for example in high_loss_high_correct:
    print(f"Loss: {example['top_ranking_passages_average_loss']}, Correct Top Ranking: {example['correct_top_ranking']}")

print("\nLow loss and low correct_top_ranking examples:")
for example in low_loss_low_correct:
    print(f"Loss: {example['top_ranking_passages_average_loss']}, Correct Top Ranking: {example['correct_top_ranking']}")
