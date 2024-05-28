import pickle
import torch
from src.models.flmr.models.flmr.modeling_flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
)


prefixes = [
    "Using the provided image, obtain documents that address the subsequent question: ",
    "Retrieve documents that provide an answer to the question alongside the image: ",
    "Extract documents linked to the question provided in conjunction with the image: ",
    "Utilizing the given image, obtain documents that respond to the following question: ",
    "Using the given image, access documents that provide insights into the following question: ",
    "Obtain documents that correspond to the inquiry alongside the provided image: ",
    "With the provided image, gather documents that offer a solution to the question: ",
    "Utilizing the given image, obtain documents that respond to the following question: ",
]

with open("sample_batched.pkl", 'rb') as f:
    batch = pickle.load(f)


pretrain_config = FLMRConfig.from_pretrained(
    "LinWeizheDragon/PreFLMR_ViT-B", trust_remote_code=True
)

query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
    "LinWeizheDragon/PreFLMR_ViT-B", subfolder="query_tokenizer"
)
context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
    "LinWeizheDragon/PreFLMR_ViT-B", subfolder="context_tokenizer"
)

instruction_token_id = query_tokenizer.encode(
                pretrain_config.mask_instruction_token, add_special_tokens=False
            )[0]

def remove_prefix(text, prefixes):
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text

def prepare_decoder_inputs(query_text_sequences, context_text_sequences, tokenizer, max_query_length, max_context_length, max_decoder_source_length, docs_per_query):
    # Tokenize and truncate query sequences
    truncated_query = [
        tokenizer.decode(tokenizer.encode(
            input_text, add_special_tokens=False, 
            max_length=max_query_length, truncation=True
        )) for input_text in query_text_sequences
    ]

    # Tokenize and truncate context sequences
    truncated_context = [
        tokenizer.decode(tokenizer.encode(
            context_text, add_special_tokens=False, 
            max_length=max_context_length, truncation=True
        )) for context_text in context_text_sequences
    ]

    concatenated_sequences = []

    # Concatenate sequences using the provided prompt template function
    for i, input_text in enumerate(truncated_query):
        for j in range(docs_per_query):
            context_index = i * docs_per_query + j
            context_text = truncated_context[context_index]
            concatenated_sequences.append((input_text, context_text))


    # Process the concatenated sequences into the desired input format
    inputs = tokenizer.batch_encode_plus(
        concatenated_sequences, 
        add_special_tokens=True,
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=max_decoder_source_length,
        return_attention_mask=True
    )

    return inputs

input_text_sequences = [remove_prefix(s, prefixes) for s in batch['input_text_sequences']]
output = prepare_decoder_inputs(input_text_sequences, batch['decoder_input_text_sequences'], query_tokenizer, 30, 470, 512, 5)

# query_tokenizer.decode(output['input_ids'][4])

# import torch
# import torch.nn as nn
# from transformers import Blip2ForConditionalGeneration, Blip2Config, Blip2Processor, Blip2Model
# GENERATION_TOKEN = "<GEN>"

# # Load the configuration
# config = Blip2Config.from_pretrained('Salesforce/blip2-opt-2.7b')
# model = Blip2ForConditionalGeneration(config).to('cuda')
# processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
# processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<GEN>']})
# gen_score_id = processor.tokenizer.convert_tokens_to_ids(['<GEN>'])[0]


# # Sample input and context text sequences
# input_text_sequences = batch['input_text_sequences'][:2]
# context_text_sequences = batch['decoder_input_text_sequences'][:10]

# concatenated_sequences = []
# truncated_query = [
#     processor.tokenizer.decode(processor.tokenizer.encode(
#         f"Query: {'aaaa'*1000}", add_special_tokens=False, 
#         max_length=32, truncation=True
#     )) for input_text in input_text_sequences
# ]

# truncated_context = [
#     processor.tokenizer.decode(processor.tokenizer.encode(
#         f"Document: {'aaaa'*1000}", add_special_tokens=False, 
#         max_length=476, truncation=True
#     )) for context_text in context_text_sequences
# ]

# for i, input_text in enumerate(truncated_query):
#     for j in range(5):
#         context_index = i * 5 + j
#         context_text = truncated_context[context_index]
#         concatenated_sequence = f"{input_text} {context_text} Relevant:"
#         concatenated_sequences.append(concatenated_sequence)

# inputs = processor(
#     text=concatenated_sequences, 
#     return_tensors="pt", 
#     padding="longest", 
#     truncation=True, 
#     max_length= 512
# )

# # # Verify the shapes
# print("Inputs shape:", inputs.input_ids.shape)

# # processor.tokenizer.convert_tokens_to_ids(['Relevant:'])[0]
# # processor.tokenizer.encode('Query:', add_special_tokens=False)

# # model.eval()

# # # Move inputs and labels to the correct device
# # input_ids = inputs.input_ids.to(model.device)
# # attention_mask = inputs.attention_mask.to(model.device)
# # pixel_values = batch['pixel_values'][:2].repeat_interleave(5, 0).to(model.device)
# # rel_score = nn.Linear(config.text_config.hidden_size, 1, bias=False).to(model.device)

# # # Forward pass with evaluation mode and no gradient tracking
# # with torch.no_grad():
# #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, output_hidden_states = True)
# #     hidden_states = outputs.language_model_outputs.hidden_states[-1]
# #     rel_position = (torch.eq(input_ids, gen_score_id).long().argmax(-1)).to(hidden_states.device)
# #     batch_size = hidden_states.size()[0]
# #     rel_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), rel_position]
# #     rel_logits = rel_score(rel_hidden_states)
    
    
# # # # Obtain the token ids for 'yes' and 'no'
# # # yes_token_id = processor.tokenizer.convert_tokens_to_ids('yes')
# # # no_token_id = processor.tokenizer.convert_tokens_to_ids('no')

# # # # Extract logits for 'yes' and 'no' tokens
# # # yes_logits = logits[..., yes_token_id]
# # # no_logits = logits[..., no_token_id]

# # # # Stack the logits and apply softmax to get probabilities
# # # stacked_logits = torch.stack([yes_logits, no_logits], dim=-1)
# # # probabilities = torch.nn.functional.softmax(stacked_logits, dim=-1)

# # # # Extract the probabilities for the 'yes' token
# # # yes_probabilities = probabilities[..., 0]

# # # # Print the probabilities for 'yes'
# # # print("Yes Probabilities:", yes_probabilities)

# # # # Output the loss
# # # print("Loss:", outputs.loss.item())

