import json
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import statistics


POS = 0
NEG = 0

import json
import math
import statistics
from collections import defaultdict
import numpy as np

# Assuming high_loss_high_correct and low_loss_low_correct are defined somewhere in the code

# for i in [3021, 5035, 7049, 9063, 11077, 13091, 15105, 17119]:
#     print(f"Results for {i}")
#     RERANKER_RESULTS = f"/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Interaction_MORES_5_B_ckpt_model_step_{i}/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"
RERANKER_RESULTS = f"/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMRQuery_Full_Context_Retrieved_Rerank_ckpt_model_step_1000/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"

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
    global POS
    global NEG
    POS, NEG = 0, 0
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
                if label == 1:
                    POS += 1
                else:
                    NEG += 1
                
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
            
            # Calculate average loss
            total_loss = sum(passage['loss'] for passage in passages)
            average_loss = total_loss / len(passages)
            item[f'{key}_average_loss'] = average_loss
            
            # Calculate average score
            total_score = sum(passage['score'] for passage in passages)
            average_score = total_score / len(passages)
            item[f'{key}_average_score'] = average_score
            
            # Calculate standard deviation of scores
            scores = [passage['score'] for passage in passages]
            std_score = statistics.stdev(scores)
            item[f'{key}_std_score'] = std_score
            
            # Calculate median of scores
            median_score = statistics.median(scores)
            item[f'{key}_median_score'] = median_score
        
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

# Calculate overall average score, median score, and std score
def calculate_overall_scores(data):
    total_score = 0
    scores = []

    for item in data:
        total_score += sum(passage['score'] for passage in item['top_ranking_passages'])
        scores.extend([passage['score'] for passage in item['top_ranking_passages']])

    overall_average_score = total_score / len(scores) if scores else 0
    overall_median_score = statistics.median(scores) if scores else 0
    overall_std_score = statistics.stdev(scores) if len(scores) > 1 else 0

    return overall_average_score, overall_median_score, overall_std_score

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
        average_score = item['top_ranking_passages_average_score']
        std_score = item['top_ranking_passages_std_score']
        median_score = item['top_ranking_passages_median_score']
        
        grouped_data[correct_top_ranking].append({
            'loss': average_loss,
            'score': average_score,
            'std_score': std_score,
            'median_score': median_score
        })
    
    average_metrics_by_correct_top_ranking = {}
    
    for correct_top_ranking, metrics in grouped_data.items():
        average_metrics_by_correct_top_ranking[correct_top_ranking] = {
            'average_loss': np.mean([m['loss'] for m in metrics]),
            'median_loss': np.median([m['loss'] for m in metrics]),
            'average_score': np.mean([m['score'] for m in metrics]),
            'median_score': np.median([m['score'] for m in metrics]),
            'std_score': np.mean([m['std_score'] for m in metrics]),
            'count': len(metrics)
        }
    
    return average_metrics_by_correct_top_ranking

# Step 1: Modify passages and calculate loss
modified_data = modify_passages_and_calculate_loss(data)

# Step 2: Calculate average loss per item
data_with_avg_loss = calculate_average_loss_per_item(modified_data)

# Step 3: Calculate overall average loss across all items
overall_top_avg_loss = calculate_overall_average_loss(data_with_avg_loss)

# Print the overall average losses
print("Overall average loss for top_ranking_passages:", overall_top_avg_loss)

# # Calculate overall average score, median score, and std score
# overall_avg_score, overall_median_score, overall_std_score = calculate_overall_scores(data_with_avg_loss)

# # Print the overall average scores
# print("\nOverall scores for top_ranking_passages:")
# print(f"Overall average score: {overall_avg_score}")
# print(f"Overall median score: {overall_median_score}")
# print(f"Overall std score: {overall_std_score}")

# # Step 4: Group by correct top ranking and calculate average and median loss
# average_metrics_by_correct_top_ranking = group_by_correct_top_ranking_and_calculate_average_loss(data_with_avg_loss)

# # Print the average and median losses by correct top ranking
# print("\nAverage and median loss by correct_top_ranking:")
# for correct_top_ranking, metrics in sorted(average_metrics_by_correct_top_ranking.items()):
#     avg_loss = metrics['average_loss']
#     median_loss = metrics['median_loss']
#     avg_score = metrics['average_score']
#     median_score = metrics['median_score']
#     std_score = metrics['std_score']
#     count = metrics['count']
#     print(f"Correct Top Ranking: {correct_top_ranking}, Average Loss: {avg_loss}, Median Loss: {median_loss}, Count: {count}")
#     print(f"Correct Top Ranking: {correct_top_ranking}, Average Score: {avg_score}, Median Score: {median_score}, Std Score: {std_score}")

# # Example usage of finding examples with high and low loss
# high_loss_high_correct, low_loss_low_correct = find_examples_with_high_low_loss(data, threshold=1, cutoff=3)


# # Calculate the minimum and maximum loss for high and low loss examples
# high_loss_high_correct_min_max = [(min([p['loss'] for p in item['top_ranking_passages']]), max([p['loss'] for p in item['top_ranking_passages']])) for item in high_loss_high_correct]
# low_loss_low_correct_min_max = [(min([p['loss'] for p in item['top_ranking_passages']]), max([p['loss'] for p in item['top_ranking_passages']])) for item in low_loss_low_correct]

# print("\nHigh loss and high correct_top_ranking examples:")
# for example, (min_loss, max_loss) in zip(high_loss_high_correct, high_loss_high_correct_min_max):
#     print(f"Loss: {example['top_ranking_passages_average_loss']}, Average Score: {example['top_ranking_passages_average_score']}, Std Score: {example['top_ranking_passages_std_score']}, Median Score: {example['top_ranking_passages_median_score']}, Correct Top Ranking: {example['correct_top_ranking']}, Min Loss: {min_loss}, Max Loss: {max_loss}")

# print("\nLow loss and low correct_top_ranking examples:")
# for example, (min_loss, max_loss) in zip(low_loss_low_correct, low_loss_low_correct_min_max):
#     print(f"Loss: {example['top_ranking_passages_average_loss']}, Average Score: {example['top_ranking_passages_average_score']}, Std Score: {example['top_ranking_passages_std_score']}, Median Score: {example['top_ranking_passages_median_score']}, Correct Top Ranking: {example['correct_top_ranking']}, Min Loss: {min_loss}, Max Loss: {max_loss}")
    
    
# # High loss and high correct_top_ranking examples
# high_loss_high_correct_losses = [p['loss'] for item in high_loss_high_correct for p in item['top_ranking_passages']]
# high_loss_high_correct_scores = [p['score'] for item in high_loss_high_correct for p in item['top_ranking_passages']]

# plt.figure()
# plt.hist(high_loss_high_correct_losses, bins=10, color='blue', edgecolor='black')
# plt.title('Histogram of Losses - High Loss High Correct')
# plt.xlabel('Loss')
# plt.ylabel('Frequency')
# plt.savefig('high_loss_high_correct_histogram.png')
# plt.close()

# # Scatter plot for score vs loss for high loss high correct
# plt.figure()
# plt.scatter(high_loss_high_correct_scores, high_loss_high_correct_losses, color='blue', edgecolor='black')
# plt.title('Score vs Loss - High Loss High Correct')
# plt.xlabel('Score')
# plt.ylabel('Loss')
# plt.savefig('high_loss_high_correct_scatter.png')
# plt.close()

# # Low loss and low correct_top_ranking examples
# low_loss_low_correct_losses = [p['loss'] for item in low_loss_low_correct for p in item['top_ranking_passages']]
# low_loss_low_correct_scores = [p['score'] for item in low_loss_low_correct for p in item['top_ranking_passages']]

# plt.figure()
# plt.hist(low_loss_low_correct_losses, bins=10, color='red', edgecolor='black')
# plt.title('Histogram of Losses - Low Loss Low Correct')
# plt.xlabel('Loss')
# plt.ylabel('Frequency')
# plt.savefig('low_loss_low_correct_histogram.png')
# plt.close()

# # Scatter plot for score vs loss for low loss low correct
# plt.figure()
# plt.scatter(low_loss_low_correct_scores, low_loss_low_correct_losses, color='red', edgecolor='black')
# plt.title('Score vs Loss - Low Loss Low Correct')
# plt.xlabel('Score')
# plt.ylabel('Loss')
# plt.savefig('low_loss_low_correct_scatter.png')
# plt.close()

print(POS)
print(NEG)