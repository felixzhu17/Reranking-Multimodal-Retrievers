# from datasets import load_dataset

# # Define the dataset and config
# dataset_name = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
# config_name = "EVQA_data"

# # Load the full dataset
# dataset = load_dataset(dataset_name, config_name)

# # Access the train split
# train_data = dataset['train']

# # Print the structure and length of the dataset
# print(f"Number of samples in train split: {len(train_data)}")
# print(train_data.column_names)

# import os

# # Directory to save the dataset
# save_dir = "offline_data"
# os.makedirs(save_dir, exist_ok=True)

# # Optionally, save as CSV
# train_data.to_csv(os.path.join(save_dir, "EVQA_Train_Full_Data.csv"))

# from datasets import load_dataset
# import os

# # Define the dataset and config
# dataset_name = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
# config_name = "OKVQA_data"

# # Load the full dataset
# dataset = load_dataset(dataset_name, config_name)

# # Access the test split
# test_data = dataset['test']

# # Print the structure and length of the test dataset
# print(f"Number of samples in test split: {len(test_data)}")
# print(test_data.column_names)

# # Directory to save the dataset
# save_dir = "offline_data"
# os.makedirs(save_dir, exist_ok=True)

# # Save the test split to a CSV file
# test_data.to_csv(os.path.join(save_dir, "OKVQA_Test_Full_Data.csv"))

# import os
# from datasets import load_dataset
# import json

# # Load the dataset
# dataset = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "OKVQA_passages")

# # Access the 'test_passages' split
# test_passages = dataset['test_passages']
# # save_dir = "offline_data"
# test_passages.to_json(os.path.join(save_dir, "OKVQA_Passages.json"))