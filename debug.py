import pickle
import json
questionId2topPassages = {}
with open("/home/fz288/rds/hpc-work/PreFLMR/search_index/EVQA/PreFLMR-G/_test_EVQADatasetForDPR.test_predictions_rank_0.pkl", "rb") as f:
    predictions = pickle.load(f)["output"]
    for pred in predictions:
        q_id = pred["question_id"]
        top_ranking_passages = pred["top_ranking_passages"]
        questionId2topPassages[q_id] = top_ranking_passages

