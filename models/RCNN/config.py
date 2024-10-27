from torch.cuda import is_available
from clib.utils import Options

class TestOptions(Options):
    def __init__(self):
        super().__init__('LeNet')
        self.update({
            'device': 'cuda' if is_available() else 'cpu',
            'model_path': 'path/to/model.pth',
            'dataset_path': 'path/to/dataset',
            'batch_size': 8,
            'num_classes': 10,
            'use_relu': False,
            'use_max_pool': False,
            'comment': ''
        })