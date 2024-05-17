import pickle


with open("sample_batched.pkl", 'rb') as f:
    batch = pickle.load(f)


import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Config, Blip2Processor


class Blip2ForRear(Blip2ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.rel_score = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        input_ids = kwargs.get('input_ids')


        hidden_states = outputs.last_hidden_state
        rel_position = (torch.eq(input_ids, self.config.gen_score_id).long().argmax(-1)).to(hidden_states.device)
        batch_size = hidden_states.size()[0]
        rel_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), rel_position]
        rel_logits = self.rel_score(rel_hidden_states)
        return rel_logits


class DecoderHeadRerankModel(nn.Module):
    def __init__(self, config):
        super(DecoderHeadRerankModel, self).__init__()
        self.model = Blip2ForConditionalGeneration(config).to('cuda')
    def forward():
        


# Load the configuration
config = Blip2Config.from_pretrained('Salesforce/blip2-flan-t5-xl')
model = Blip2ForConditionalGeneration(config).to('cuda')
processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')

# Sample input and context text sequences
input_text_sequences = batch['input_text_sequences']
context_text_sequences = batch['decoder_input_text_sequences']

# Concatenate sequences and create labels
concatenated_sequences = []
labels = []
for i, input_text in enumerate(input_text_sequences):
    for j in range(5):
        context_index = i * 5 + j
        context_text = context_text_sequences[context_index]
        concatenated_sequence = f"Query: {input_text} Document: {context_text} Relevant:"
        concatenated_sequences.append(concatenated_sequence)
        # First of each group of 5 gets 'yes', the others 'no'
        if j == 0:
            labels.append("yes")
        else:
            labels.append("no")

# Tokenize the concatenated sequences and targets
inputs = processor(text=concatenated_sequences, return_tensors="pt", padding=True, truncation=True)
target_tokens = processor(text=labels, return_tensors="pt", padding=True, truncation=True).input_ids
labels = target_tokens.reshape(-1, 2) 

# Verify the shapes
print("Inputs shape:", inputs.input_ids.shape)
print("Labels shape:", labels.shape)

model.eval()

# Move inputs and labels to the correct device
inputs = {key: val.to(model.device) for key, val in inputs.items()}
inputs['pixel_values'] = batch['pixel_values'].repeat_interleave(5,0).to(model.device)
labels = labels.to(model.device)

# Forward pass with evaluation mode and no gradient tracking
with torch.no_grad():
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits[:,0,:]

# Obtain the token ids for 'yes' and 'no'
yes_token_id = processor.tokenizer.convert_tokens_to_ids('yes')
no_token_id = processor.tokenizer.convert_tokens_to_ids('no')

# Extract logits for 'yes' and 'no' tokens
yes_logits = logits[..., yes_token_id]
no_logits = logits[..., no_token_id]

# Stack the logits and apply softmax to get probabilities
stacked_logits = torch.stack([yes_logits, no_logits], dim=-1)
probabilities = torch.nn.functional.softmax(stacked_logits, dim=-1)

# Extract the probabilities for the 'yes' token
yes_probabilities = probabilities[..., 0]

# Print the probabilities for 'yes'
print("Yes Probabilities:", yes_probabilities)

# Output the loss
print("Loss:", outputs.loss.item())
