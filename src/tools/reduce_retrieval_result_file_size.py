import json
from tqdm import tqdm

input_files = [
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMR_Index_2/test/_test_OKVQADatasetForDPR.test_predictions_rank_0.json",
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/TEST_OKVQA_FLMR_Index/test/_test_OKVQADatasetForDPR.train_predictions_rank_0.json",
]

for file_path in input_files:
    print(file_path)
    target_file_path = file_path.replace(".json", ".pkl")

    with open(file_path, "r") as f:
        data = json.load(f)

    # data = data['output']
    new_data = {
        "output": [],
    }
    for prediction in tqdm(data["output"]):
        # print(prediction.keys())
        for pred in prediction["top_ranking_passages"]:
            pred.pop("content")
        new_data["output"].append(prediction)

    # save new data to target_file_path
    print(f"Saving to {target_file_path}")
    # save as pickle
    import pickle

    with open(target_file_path, "wb") as f:
        pickle.dump(new_data, f)
