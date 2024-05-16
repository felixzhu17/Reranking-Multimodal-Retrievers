import pickle


with open("sample_batched.pkl", 'rb') as f:
    batch = pickle.load(f)


from transformers import Blip2ForConditionalGeneration, Blip2Config, Blip2Processor

# Load the configuration
config = Blip2Config.from_pretrained('Salesforce/blip2-flan-t5-xl')
model = Blip2ForConditionalGeneration(config)
tokenizer = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')


# Sample input and context text sequences
input_text_sequences = batch['input_text_ids']
context_text_sequences = batch['context_text_ids']

# Concatenate sequences and create labels
concatenated_sequences = []
labels = []
for i, input_text in enumerate(input_text_sequences):
    for j in range(5):
        context_index = i * 5 + j
        context_text = context_text_sequences[context_index]
        concatenated_sequence = input_text + " " + context_text
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

# # Align the labels for token prediction:
# # Fill other parts with -100 (to ignore them in loss computation)
# labels = torch.full_like(inputs.input_ids, -100)
# labels[:, -1] = target_tokens[:, 0]  # Only the last token is the target

labels = target_tokens[:, 0].reshape(-1, 1) 

# Verify the shapes
print("Inputs shape:", inputs.input_ids.shape)
print("Labels shape:", labels.shape)

# Generate fake pixel values
batch_size = inputs.input_ids.size(0)
num_channels = 3  # Typically, images have 3 channels (RGB)
height = 224  # Common height for vision models
width = 224  # Common width for vision models
fake_pixel_values = torch.randn((batch_size, num_channels, height, width))



model.eval()


# Move inputs and labels to the correct device
inputs = {key: val.to(model.device) for key, val in inputs.items()}
labels = labels.to(model.device)
fake_pixel_values = fake_pixel_values.to(model.device)

# Add fake pixel values to the inputs dictionary
inputs['pixel_values'] = fake_pixel_values

# Forward pass with evaluation mode and no gradient tracking
with torch.no_grad():
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
# Output the loss
print("Loss:", loss.item())