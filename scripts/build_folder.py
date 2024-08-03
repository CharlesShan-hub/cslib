from pathlib import Path
import click

def build_file(path, content=[]):
    with open(path, 'w') as f:
        for line in content:
            if line is None:
                continue
            f.write(line+'\n')

@click.command()
@click.option('--name','-n', help='Name of algorithm.')
@click.option('--path','-p', help='Base path for the block.')
@click.option('--title','-t', help='Title of the paper.')
@click.option('--link','-l', help='Link to the paper.')
@click.option('--arxiv','-v', help='ArXiv of the paper.')
@click.option('--author','-a', help='Modified from ... or self written.')
def main(name,path,title,link,arxiv,author):
    print(f'***********************************************')
    print(f'*     Build new block for a new algorithm     *')
    print(f'***********************************************')
    
    base_path = Path(path)
    if base_path.exists() == False:
        print(f'❌ Path: {base_path} not exist! Build Failed!')
        print('Please input path to your CVPlayground!')
        return
    print(f'✅ Path: {base_path}')

    model_path = Path(base_path,'src','clib','model','collection',name)
    if model_path.exists():
        print(f'❌ Name: {name} has build! Change a name!')
        return
    print(f'✅ Name: {name}')
    model_path.mkdir()

    print(f'✅ Title: {title}')
    print(f'✅ Link: {link}')
    print(f'✅ ArXiv: {arxiv}')
    print(f'✅ Author: {author}')
    build_file(Path(model_path,"__init__.py"),
        [
            f'"""',
            f'    {title}',
            f'    Paper: {link}',
            f'    ArXiv: {arxiv}' if arxiv != "" else None,
            f'    Modified from: {author}' if author != "" else '    Author: Charles Shan',
            f'"""',
            f'from .model import {name} as Model, load_model',
            f'from .inference import inference',
            f'from .train import train'
        ])
    build_file(Path(model_path,"config.py"),
        [
            f'from torch.cuda import is_available',
            f'from ....utils import Options',
            f'',
            f'',
            f'class TrainOptions(Options):',
            f'    """',
            f'                                                    Argument Explaination',
            f'        ======================================================================================================================',
            f'                Symbol          Type            Default                         Explaination',
            f'        ----------------------------------------------------------------------------------------------------------------------',
            f'            --pre_trained       Str            model.pth                     The path of pre-trained model',
            f'        ----------------------------------------------------------------------------------------------------------------------',
            f'    """',
            f'    def __init__(self):',
            f"        super().__init__('{name}')",
             '        self.update({',
            f"            'pre_trained': 'model.pth',",
            f"            'device': 'cuda' if is_available() else 'cpu',",
            f"            'dataset_path': '../../data/mnist', ",
            f"            'epochs': 200, ",
            f"            'batch_size': 64, ",
            f"            'lr': 0.0002, ",
             "        })",
            f'',
            f'',
            f'class TestOptions(Options):',
            f'    """',
            f'                                                    Argument Explaination',
            f'        ======================================================================================================================',
            f'                Symbol          Type            Default                         Explaination',
            f'        ----------------------------------------------------------------------------------------------------------------------',
            f'            --pre_trained       Str            model.pth                     The path of pre-trained model',
            f'        ----------------------------------------------------------------------------------------------------------------------',
            f'    """',
            f'    def __init__(self):',
            f"        super().__init__('{name}')",
             '        self.update({',
            f"            'pre_trained': 'model.pth',",
            f"            'device': 'cuda' if is_available() else 'cpu',",
             "        })",
        ])
    build_file(Path(model_path,"train.py"),
        [
            f'from tqdm import tqdm',
            f'import torch',
            f'import torch.nn as nn',
            f'import torch.optim as optim',
            f'from torchvision import datasets, transforms',
            f'from torch.utils.data import DataLoader\n',
            f'from .config import TrainOptions',
            f'from .model import {name} as Model\n'
             'def train(opts={}):',
            f'    # Init',
            f'    opts = TrainOptions().parse(opts,present=False)\n',
            f'    model = Model().to(opts.device)\n',
            f'    if criterion is None:',
            f'        pass # criterion = nn.CrossEntropyLoss()\n',
            f'    if optimizer is None:',
            f'        pass # optimizer = optim.Adam(model.parameters(), lr=opts.lr)\n',
            f'    transform = transforms.Compose([',
            f'        transforms.Resize((28, 28)),',
            f'        transforms.ToTensor(),',
            f'        transforms.Normalize((0.5,), (0.5,))',
            f'    ])\n',
            f'    if train_loader is None and test_loader is None:',
             "        delattr(opts, 'dataset_path')",
            f'        train_dataset = datasets.MNIST(root=opts.TorchVisionPath, train=True, download=True, transform=transform)',
            f'        test_dataset = datasets.MNIST(root=opts.TorchVisionPath, train=False, download=True, transform=transform)',
            f'        train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True)',
            f'        test_loader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False)',
            f'    ',
            f'    # Train',
            f'    opts.presentParameters()',
            f'    for epoch in range(opts.epochs):',
            f'        model.train()',
            f'        running_loss = 0.0',
            f'        pbar = tqdm(train_loader, total=len(train_loader))',
            f'        for images, labels in pbar:',
            f'            images, labels = images.to(opts.device), labels.to(opts.device)',
            f'            optimizer.zero_grad()',
            f'            outputs = model(images)',
            f'            loss = criterion(outputs, labels)',
            f'            loss.backward()',
            f'            optimizer.step()',
            f'            running_loss += loss.item()',
             '            pbar.set_description(f"Epoch [{epoch+1}/{opts.epochs}]")',
            f'            pbar.set_postfix(loss=(running_loss / (epoch * len(train_loader) + 1)))',
             '        print(f"Epoch [{epoch+1}/{opts.epochs}], Loss: {running_loss/len(train_loader):.4f}")',
            f'    ',
            f'    # Test',
            f'    model.eval()',
            f'    with torch.no_grad():',
            f'        correct = 0',
            f'        total = 0',
            f'        for images, labels in test_loader:',
            f'            images, labels = images.to(opts.device), labels.to(opts.device)',
            f'            outputs = model(images)',
            f'            _, predicted = torch.max(outputs.data, 1)',
            f'            total += labels.size(0)',
            f'            correct += (predicted == labels).sum().item()',
            f'            ',
             '    print(f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%")',
        ])
    build_file(Path(model_path,"inference.py"),
        [
            f'from .config import TestOptions\n',
            'def inference(opts={}):',
            '    opts = TestOptions().parse(opts)'
        ])
    build_file(Path(model_path,"model.py"),
        [
            f'import torch',
            f'import torch.nn as nn',
            f'',
            f'def load_model(opts):',
            f'    model = {name}().to(opts.device)',
            f'    params = torch.load(opts.pre_trained, map_location=opts.device)',
            f'    model.load_state_dict(params)',
            f'    return model',
            f'',
            f'class {name}(nn.Module):',
            f'    def __init__(self):',
            f'        super({name}, self).__init__()',
            f'    ',
            f'    def forward(self, x):',
            f'        return x'
        ])
    build_file(Path(model_path,"utils.py"))


if __name__ == '__main__':
    main()