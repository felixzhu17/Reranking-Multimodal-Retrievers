import pickle

# passage_id2doc = None 

# import json
# import pickle
# from src.models.flmr import FLMRContextEncoderTokenizer

# input_files = [
#     "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_PreFLMR/test/index/index_test_OKVQADatasetForDPR.test_predictions_rank_0.json",
#     "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_PreFLMR/test/index/index_test_OKVQADatasetForDPR.train_predictions_rank_0.json",
# ]

# # tokenizer = FLMRContextEncoderTokenizer.from_pretrained("LinWeizheDragon/PreFLMR_ViT-B", subfolder="context_tokenizer")

# questionId2topPassages = {}
# for prediction_pkl in input_files:
#     if prediction_pkl.endswith('.json'):
#         # load using json
#         with open(prediction_pkl, 'r') as f:
#             predictions = json.load(f)['output']
#             for pred in predictions:
#                 q_id = pred['question_id']
#                 top_ranking_passages = pred['top_ranking_passages']
#                 questionId2topPassages[q_id] = top_ranking_passages
#     else:
#         # Can use `src/tools/reduce_retrieval_result_file_size.py` to reduce json file size to speed up the loading
#         # in this case, we load from a pkl file
#         with open(prediction_pkl, 'rb') as f:
#             predictions = pickle.load(f)['output']
#             for pred in predictions:
#                 q_id = pred['question_id']
#                 top_ranking_passages = pred['top_ranking_passages']
#                 questionId2topPassages[q_id] = top_ranking_passages

# # Assert that the length of all questionId2topPassages values are equal
# lengths = [len(passages) for passages in questionId2topPassages.values()]
# assert all(length == lengths[0] for length in lengths)

# with open("/home/fz288/rds/hpc-work/PreFLMR/question_ids.pkl", 'rb') as f:
#     questions = pickle.load(f)

with open("upper_right.pkl", 'rb') as f:
    batch = pickle.load(f)



# for q_id in questions:
#     assert q_id in questionId2topPassages


# predictions[0]['top_ranking_passages'][0]['content']

# encoding = tokenizer([predictions[0]['top_ranking_passages'][0]['content'], predictions[0]['top_ranking_passages'][1]['content']],
#                                     padding='max_length',
#                                     max_length=512,
#                                     truncation=True,
#                                     return_tensors="pt")
# generator_input_ids, generator_attention_mask = encoding.input_ids, encoding.attention_mask
# generator_input_ids = generator_input_ids.to(labels.device)
# generator_attention_mask = generator_attention_mask.to(labels.device)

