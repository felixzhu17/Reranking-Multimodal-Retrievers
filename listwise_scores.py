import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# List of steps and number of negatives
steps = [2014, 4028, 6042, 8056, 10070]
# num_negs_list = [4, 9, 19, 49]
num_negs_list = [4]
base_path = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Interaction_MORES_5_B_Neg_Sample_ckpt_model_step_"

# Function to calculate cross-entropy loss
def calculate_cross_entropy_loss(pos_score, all_scores):
    exp_scores = [math.exp(score) for score in all_scores]
    sum_exp_scores = sum(exp_scores)
    exp_pos_score = math.exp(pos_score)
    return -math.log(exp_pos_score / sum_exp_scores)

# Function to calculate correctness based on passage_id
def calculate_correctness(passage_list, pos_item_ids):
    return min(1, sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids))

# Function to calculate recall
def calculate_recall(grouped_data):
    correct_count = len(grouped_data[1])
    incorrect_count = len(grouped_data[0])
    recall = correct_count / (incorrect_count + correct_count) if (incorrect_count + correct_count) > 0 else None
    return correct_count, incorrect_count, recall

# Function to sample passages and calculate cross-entropy loss
def sample_passages_and_calculate_loss(data, num_negs, num_samples=500):
    total_losses_correct = []
    total_losses_incorrect = []
    queries_with_pos_passage = 0
    queries_without_pos_passage = 0
    all_scores_correct = []
    all_scores_incorrect = []

    for item in data:
        pos_item_ids = set(item['pos_item_ids'])
        top_ranking_passages = item['top_ranking_passages']
        pos_passages = [p for p in top_ranking_passages if p['passage_id'] in pos_item_ids]
        neg_passages = [p for p in top_ranking_passages if p['passage_id'] not in pos_item_ids]

        if len(pos_passages) > 0:
            queries_with_pos_passage += 1
            all_scores = [p['score'] for p in top_ranking_passages]
            if calculate_correctness(top_ranking_passages[:5], pos_item_ids) == 1:
                all_scores_correct.extend(all_scores)
            else:
                all_scores_incorrect.extend(all_scores)

            if len(neg_passages) >= num_negs:
                cross_entropy_losses = []
                for _ in range(num_samples):
                    selected_pos = random.choice(pos_passages)
                    selected_neg = random.sample(neg_passages, num_negs)
                    selected_passages = [selected_pos] + selected_neg

                    pos_score = selected_pos['score']
                    all_scores_sampled = [p['score'] for p in selected_passages]

                    # Calculate cross-entropy loss
                    loss = calculate_cross_entropy_loss(pos_score, all_scores_sampled)
                    cross_entropy_losses.append(loss)

                # Average cross-entropy loss for this query
                average_loss = np.mean(cross_entropy_losses)

                # Calculate correctness once if cross-entropy loss is calculated
                correctness = calculate_correctness(top_ranking_passages[:5], pos_item_ids)
                item['correctness'] = correctness

                if correctness == 1:
                    total_losses_correct.append(average_loss)
                else:
                    total_losses_incorrect.append(average_loss)
            else:
                item['correctness'] = None
        else:
            queries_without_pos_passage += 1
            item['correctness'] = None

    return total_losses_correct, total_losses_incorrect, data, queries_with_pos_passage, queries_without_pos_passage, all_scores_correct, all_scores_incorrect

# Calculate overall average loss and recall
def calculate_overall_average_loss_and_recall(total_losses_correct, total_losses_incorrect, data):
    overall_top_average_loss_correct = np.mean(total_losses_correct) if total_losses_correct else 0
    overall_top_average_loss_incorrect = np.mean(total_losses_incorrect) if total_losses_incorrect else 0
    grouped_data = defaultdict(list)

    for item in data:
        grouped_data[item['correctness']].append(item)

    correct_count, incorrect_count, recall = calculate_recall(grouped_data)

    return overall_top_average_loss_correct, overall_top_average_loss_incorrect, recall

# Plot and save the distribution of scores
def plot_and_save_distribution(all_scores_correct, all_scores_incorrect, step, num_negs):
    plt.figure(figsize=(12, 6))

    plt.hist(all_scores_correct, bins=50, alpha=0.5, label='Correct', color='blue')
    plt.hist(all_scores_incorrect, bins=50, alpha=0.5, label='Incorrect', color='red')

    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Scores (Step: {step}, NUM_NEGS: {num_negs})')
    plt.legend()

    filename = f'distribution_scores_step_{step}_num_negs_{num_negs}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Saved distribution plot to {filename}')

# Calculate and print statistics
def print_statistics(scores, label):
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        print(f"{label} - Mean: {mean_score}, Std: {std_score}, Median: {median_score}")
    else:
        print(f"{label} - No data available")

# Loop through each combination of step and num_negs, process the data, and print the overall average loss and recall
for num_negs in num_negs_list:
    print(f"Processing with NUM_NEGS = {num_negs}")
    for step in steps:
        file_path = f"{base_path}{step}/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"

        with open(file_path, "rb") as f:
            data = json.load(f)['output']

        # Step 1: Sample passages and calculate loss
        (total_losses_correct, total_losses_incorrect, modified_data, queries_with_pos_passage, 
         queries_without_pos_passage, all_scores_correct, all_scores_incorrect) = sample_passages_and_calculate_loss(data, num_negs)

        # Step 2: Calculate overall average loss and recall across all items
        overall_top_avg_loss_correct, overall_top_avg_loss_incorrect, recall = calculate_overall_average_loss_and_recall(total_losses_correct, total_losses_incorrect, modified_data)

        # Print the overall average losses and recall
        print(f"Overall average loss for step {step} with NUM_NEGS = {num_negs}: Correct = {overall_top_avg_loss_correct}, Incorrect = {overall_top_avg_loss_incorrect}, Recall: {recall}")
        # print(f"Queries with at least one positive passage: {queries_with_pos_passage}, Queries without any positive passage: {queries_without_pos_passage}")

        # Print statistics for correct and incorrect losses
        print_statistics(total_losses_correct, "Correct Cross-Entropy Loss")
        print_statistics(total_losses_incorrect, "Incorrect Cross-Entropy Loss")

        # Print statistics for correct and incorrect scores
        print_statistics(all_scores_correct, "Correct Scores")
        print_statistics(all_scores_incorrect, "Incorrect Scores")

        # Step 3: Plot and save the distribution of scores
        plot_and_save_distribution(all_scores_correct, all_scores_incorrect, step, num_negs)
