import pickle
# # from src.models.flmr.models.flmr.configuration_flmr import FLMRConfig

# print(1)
# # Read RAG_config.pkl
# with open('RAG_config.pkl', 'rb') as file:
#     rag_config = pickle.load(file)

# # print(1)
# # # Read PreFLMR_config.pkl
# # with open('PreFLMR_config.pkl', 'rb') as file:
# #     preflmr_config = pickle.load(file)
# # print(1)
# # Read PreFLMR_train_batch.pkl
# with open('PreFLMR_train_batch.pkl', 'rb') as file:
#     preflmr_train_batch = pickle.load(file)
    
# # >>> preflmr_train_batch['query_input_ids'].shape
# # torch.Size([8, 32])
# # >>> preflmr_train_batch['query_attention_mask'].shape
# # torch.Size([8, 32])
# # >>> preflmr_train_batch['context_input_ids'].shape
# # torch.Size([40, 512])
# # >>> preflmr_train_batch['num_negative_examples']
# # 4
# # >>> preflmr_train_batch['query_pixel_values'].shape
# # torch.Size([8, 3, 224, 224])

# query_input_ids = preflmr_train_batch['query_input_ids']
# query_attention_mask = preflmr_train_batch['query_attention_mask']
# context_input_ids = preflmr_train_batch['context_input_ids']
# query_pixel_values = preflmr_train_batch['query_pixel_values']
# n_docs = preflmr_train_batch['num_negative_examples']



with open('prepared_data.pkl', 'rb') as file:
    data = pickle.load(file)