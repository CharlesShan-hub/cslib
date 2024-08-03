from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import TestOptions
from .model import load_model

def inference(opts = {}, model = None, test_loader = None):
    # Init
    opts = TestOptions().parse(opts,present=False)
    
    model = load_model(opts) # LeNet
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if test_loader is None:
        test_dataset = datasets.MNIST(root=opts.TorchVisionPath, train=False, download=True, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False)

    # Test
    opts.presentParameters()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pbar = tqdm(test_loader, total=len(test_loader))
        for images, labels in pbar:
            images, labels = images.to(opts.device), labels.to(opts.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%")
