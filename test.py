import torch
import torch.nn as nn

# Example logits and labels
logits = torch.tensor([[1.0, 2.0], [1.5, 0.5]])
labels = torch.tensor([1, 0])

# Cross-Entropy Loss
loss_fn = nn.CrossEntropyLoss()
loss_ce = loss_fn(logits, labels)
print('Cross-Entropy Loss:', loss_ce.item())

# Convert logits to probabilities using softmax
probabilities = torch.softmax(logits, dim=1)[:, 1]

# Convert labels to floats
labels_float = labels.float()

# Binary Cross-Entropy Loss
bce_loss_fn = nn.BCELoss()
loss_bce = bce_loss_fn(probabilities, labels_float)
print('Binary Cross-Entropy Loss:', loss_bce.item())