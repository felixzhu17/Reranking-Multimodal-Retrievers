import pickle


with open("sample_batched.pkl", 'rb') as f:
    batch = pickle.load(f)


from transformers import Blip2ForConditionalGeneration, Blip2Config, Blip2Processor

# Load the configuration
config = Blip2Config.from_pretrained('Salesforce/blip2-flan-t5-xl')
model = Blip2ForConditionalGeneration(config).to('cuda')
tokenizer = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')


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

from transformers import Blip2Processor
import torch
# Initialize the processor
processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')

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
    loss = outputs.loss
# Output the loss
print("Loss:", loss.item())