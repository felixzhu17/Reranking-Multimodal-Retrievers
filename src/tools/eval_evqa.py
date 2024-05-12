from pprint import pprint

import ast
import evaluation_utils

prediction_file = "/home/ubuntu/additional_data/experiments/AWS_EVQA_RAG_PreFLMR_BLIP2_T5xxl_K=5_LR1e-4_K_train=5_GT_in_training_evaluation/test/debug/predictions_rank_0.csv"

# This line can be obtained from HF field: "question_type"
# either templated or automatic
question_type = "automatic"

from tqdm import tqdm
import multiprocessing
from functools import partial

# read csv
import pandas as pd

df = pd.read_csv(prediction_file, sep="\t")
# print(df.columns)

# use multi-processing to speed up
import multiprocessing


# Function to process a single row of the DataFrame
def process_row(row, question_type):
    row = row[1]
    question = row[3]
    answers = row[4]
    prediction = row[6]

    answers = answers.replace("\n", "").replace("' '", "', '")
    answers = ast.literal_eval(answers)
    answers = [str(answer).strip() for answer in answers]
    try:
        score = evaluation_utils.evaluate_example(
            question,
            reference_list=answers,
            candidate=prediction,
            question_type=question_type,
        )
    except:
        score = 1.0
    print(score)
    return score


# Set the number of processes to use
num_processes = 64  # multiprocessing.cpu_count() // 2

# Create a partial function with the fixed argument (question_type)
partial_process_row = partial(process_row, question_type=question_type)

# Create a Pool of workers
with multiprocessing.Pool(processes=num_processes) as pool:
    # Use tqdm to display progress
    all_scores = list(
        tqdm(pool.imap(partial_process_row, df.iterrows(), chunksize=1), total=len(df))
    )

# Calculate and print the average score
average_score = sum(all_scores) / len(all_scores)
print(f"Average score: {average_score}")


# all_scores = []
# # iterate over rows
# for index, row in tqdm(df.iterrows(), total=len(df)):
#     question = row[3]
#     answers = row[4]
#     prediction = row[6]
#     # print(question)
#     answers = answers.replace("\n", "").replace("' '", "', '")
#     answers = ast.literal_eval(answers)
#     answers = [answer.strip() for answer in answers]
#     # print(prediction, "->", answers)
#     score = evaluation_utils.evaluate_example(
#         question,
#         reference_list=answers,
#         candidate=prediction,
#         question_type=question_type)
#     all_scores.append(score)
#     # break

# print(f"Average score: {sum(all_scores) / len(all_scores)}")
