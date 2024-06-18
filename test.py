import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
from PIL import Image
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

# Define the device to use for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the BLIP2 processor (no need to load model weights)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Initialize BLIP2 model with random weights
config = Blip2Config()
config.vocab_size = 30522  # Example vocab size
model = Blip2ForConditionalGeneration(config).to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-5)

# Create fake image data (e.g., 3-channel RGB image of size 224x224)
fake_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

# Create fake text data
input_text = "Is this a fake image?" 
desired_output_text = "yes"

# Process the fake image and text
inputs = processor(images=fake_image, text=input_text, return_tensors="pt").to(device)

# Tokenize the desired output text
labels = processor.tokenizer(desired_output_text, return_tensors="pt").input_ids.to(device)

# Training loop (simplified)
model.train()
for epoch in tqdm(range(100)):  # Example with a single epoch for demonstration
    optimizer.zero_grad()

    # Forward pass through the model to get the outputs
    outputs = model(**inputs, labels=labels)

    # Calculate the loss
    loss = outputs.loss

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Generate text based on the input (for validation purposes)
    generate_inputs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"]
    }
    generated_ids = model.generate(**generate_inputs, max_new_tokens=3)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Print the generated text and the loss
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    print(f"Generated Text: {generated_text}")

# # Extract the relevant portion after the input text
# input_text_length = len(input_text)
# if input_text_length < len(generated_text):
#     result_text = generated_text[input_text_length:].strip()
# else:
#     result_text = ""

# # Print the relevant portion
# print(f"Relevant Portion After Final Token: {result_text}")
