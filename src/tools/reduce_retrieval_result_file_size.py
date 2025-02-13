import json
from tqdm import tqdm

input_files = [
    "/home/fz288/rds/hpc-work/PreFLMR/search_index/EVQA/PreFLMR-B/EVQA_EVQA_PreFLMR_ViT-B_test.json",
    "/home/fz288/rds/hpc-work/PreFLMR/search_index/EVQA/PreFLMR-B/EVQA_EVQA_PreFLMR_ViT-B_train.json",
    "/home/fz288/rds/hpc-work/PreFLMR/search_index/EVQA/PreFLMR-B/EVQA_EVQA_PreFLMR_ViT-B_valid.json",
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
