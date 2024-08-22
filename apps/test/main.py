from clib.model import classify 
import sys
from pathlib import Path
sys.path.append(Path(__file__,'../../../scripts').resolve().__str__())
from config import opts

model_name = 'LeNet'

opts[model_name] = {
    '*ResPath': r'@ModelBasePath/LeNet/MNIST/',
    '*pre_trained': r'@ResPath/9775/model.pth'
}
opts = opts[model_name]

alg = getattr(classify,model_name)

opts = alg.TestOptions().parse(opts,present=False)
model = alg.load_model(opts)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

test_dataset = datasets.MNIST(root=opts.TorchVisionPath, train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False)

opts.presentParameters()
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
