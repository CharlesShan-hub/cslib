from torch.cuda import is_available
from clib.utils import Options

class TrainOptions(Options):
    def __init__(self):
        super().__init__('LeNet')
        self.update({
            'device': 'cuda' if is_available() else 'cpu',
            'dataset_path': '../../data/mnist', 
            'epochs': 1, 
            'batch_size': 64, 
            'lr': 0.001, 
            'repeat': 2,
            'seed': 42,
            'train_mode': ['Holdout','K-fold'][0]
        })


class TestOptions(Options):
    def __init__(self):
        super().__init__('LeNet')
        self.update({
            'model_path': 'model.pth',
            'device': 'cuda' if is_available() else 'cpu',
            'batch_size': 64, 
        })
