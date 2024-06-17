import sys
import os
sys.path.append(os.getcwd())
import libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Generate and Load Data
true_w = torch.tensor([2, -3.4])
true_b = 4.2
dataset = libs.data.LinearDataset(true_w, true_b, 1000, noise=True)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load Model, Loss and Train
model = libs.model.LinearModel(len(true_w))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
num_epochs = 100
for epoch in range(num_epochs):
    for features, labels in dataloader:
        # Forward pass
        outputs = model(features)
        
        # Calculate the loss
        loss = loss_function(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")