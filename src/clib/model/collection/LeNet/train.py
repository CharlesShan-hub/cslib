from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import TrainOptions
from .model import LeNet as Model

def train(opts = {}, model = None, criterion = None, optimizer = None,
          train_loader = None, test_loader = None):
    # Init
    opts = TrainOptions().parse(opts,present=False)
    
    model = Model(use_max_pool=False,use_relu=False).to(opts.device) # LeNet
    # model = LeNet(use_max_pool=True,use_relu=True).to(device) # Improved
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if train_loader is None or test_loader is None:
        delattr(opts, 'dataset_path')
        train_dataset = datasets.MNIST(root=opts.TorchVisionPath, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=opts.TorchVisionPath, train=False, download=True, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False)

    # Train
    opts.presentParameters()
    for epoch in range(opts.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader))
        for images, labels in pbar:
            images, labels = images.to(opts.device), labels.to(opts.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch [{epoch+1}/{opts.epochs}]")
            pbar.set_postfix(loss=(running_loss / (epoch * len(train_loader) + 1)))
        print(f"Epoch [{epoch+1}/{opts.epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(opts.device), labels.to(opts.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%")

    # Save Model
    torch.save(model.state_dict(), Path(opts.models_path,'model.pth'))
    opts.save()