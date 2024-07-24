import json
from scipy.stats import chi2, norm

# File path for the results
RESULTS_FILE = "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_Interaction_MORES_5_B_Train_on_Retrieve_ckpt_model_step_10070/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json"

# Load JSON file
def load_json(file_path):
    with open(file_path, "rb") as f:
        return json.load(f)['output']

data = load_json(RESULTS_FILE)

def calculate_correctness(passage_list, pos_item_ids):
    return min(1, sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids))

# Initialize contingency table values
a = b = c = d = 0

for entry in data:
    question_id = entry['question_id']
    pos_item_ids = entry['pos_item_ids']
    
    correct_raw_model = calculate_correctness(entry['raw_top_ranking_passages'][:5], pos_item_ids)
    correct_model_of_interest = calculate_correctness(entry['top_ranking_passages'][:5], pos_item_ids)
    
    if correct_raw_model == 1 and correct_model_of_interest == 1:
        a += 1
    elif correct_raw_model == 1 and correct_model_of_interest == 0:
        b += 1
    elif correct_raw_model == 0 and correct_model_of_interest == 1:
        c += 1
    else:
        d += 1

# Calculate McNemar's test statistic
mcnemar_statistic = (abs(b - c) - 1) ** 2 / (b + c)
p_value = chi2.sf(mcnemar_statistic, 1)

print(f"Contingency Table: a={a}, b={b}, c={c}, d={d}")
print(f"McNemar's test statistic: {mcnemar_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in accuracy between the two models.")
else:
    print("There is no significant difference in accuracy between the two models.")


# # Load JSON file
# def load_json(file_path):
#     with open(file_path, "rb") as f:
#         return json.load(f)['output']

# data = load_json(RESULTS_FILE)

# def calculate_correctness(passage_list, pos_item_ids):
#     return min(1, sum(1 for passage in passage_list if passage['passage_id'] in pos_item_ids))

# # Initialize counts
# correct_raw_model_count = 0
# correct_model_of_interest_count = 0
# total_questions = len(data)

# for entry in data:
#     pos_item_ids = entry['pos_item_ids']
    
#     correct_raw_model = calculate_correctness(entry['raw_top_ranking_passages'][:5], pos_item_ids)
#     correct_model_of_interest = calculate_correctness(entry['top_ranking_passages'][:5], pos_item_ids)
    
#     correct_raw_model_count += correct_raw_model
#     correct_model_of_interest_count += correct_model_of_interest

# # Proportions
# p1 = correct_raw_model_count / total_questions
# p2 = correct_model_of_interest_count / total_questions

# # Standard error of the difference
# se_diff = ((p1 * (1 - p1) + p2 * (1 - p2)) / total_questions) ** 0.5

# # Z-test for difference in proportions
# z_statistic = (p1 - p2) / se_diff
# p_value = 2 * (1 - norm.cdf(abs(z_statistic)))  # Two-tailed test

# print(f"Proportion of correct predictions for raw model: {p1}")
# print(f"Proportion of correct predictions for model of interest: {p2}")
# print(f"Z-test statistic: {z_statistic}")
# print(f"p-value: {p_value}")

# if p_value < 0.05:
#     print("There is a significant difference in the proportion of correct predictions between the two models.")
# else:
#     print("There is no significant difference in the proportion of correct predictions between the two models.")
