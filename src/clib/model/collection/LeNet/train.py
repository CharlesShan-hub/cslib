from collections import deque
from tqdm import tqdm
from pathlib import Path
from os import makedirs
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from .config import TrainOptions
from .model import LeNet as Model

def _init():
    pass

def _train():
    pass

def _test():
    pass

def train(opts = {}, model = None, criterion = None, optimizer = None,
          train_loader = None, test_loader = None):
    # Init
    opts = TrainOptions().parse(opts,present=False)

    torch.manual_seed(opts.seed)
    
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

    if Path(opts.models_path).exists() == False:
        makedirs(opts.models_path)
    writer = SummaryWriter(log_dir=opts.models_path)

    # Function to adjust learning rate
    def adjust_learning_rate(factor=0.1):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    # Train: repeat -> epoch
    opts.presentParameters()
    for i in range(opts.repeat):
        print(f"Starting training round {i+1}/{opts.repeat}, lr={optimizer.param_groups[0]['lr']}")

        # Reinitialize recent_losses deque for each round of training
        recent_losses = deque(maxlen=3)

        epoch = 0
        while True:
            epoch += 1
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
                pbar.set_description(f"Epoch [{epoch}/{opts.epochs if opts.epochs != -1 else '∞'}]")
                pbar.set_postfix(loss=(running_loss / (epoch * len(train_loader) + 1)))
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch}/{opts.epochs if opts.epochs != -1 else '∞'}], Loss: {epoch_loss:.4f}")

            # Log loss to TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            
            # Store the current epoch's loss
            recent_losses.append(epoch_loss)

            # Check if the loss has not decreased for three consecutive epochs
            if len(recent_losses) == 3 and all(x <= y for x, y in zip(recent_losses, list(recent_losses)[1:])):
                print("Training has converged. Stopping...")
                break

            # Stop if the number of epochs is reached (but only if opts.epochs is not -1)
            if opts.epochs != -1 and epoch >= opts.epochs:
                break
        
        # Adjust learning rate at the beginning of each round
        if i != opts.repeat-1:
            adjust_learning_rate(factor=opts.factor)
            print(f"Adjusted learning rate to {optimizer.param_groups[0]['lr']:.6f} for this round.")


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