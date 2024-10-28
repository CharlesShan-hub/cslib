from torch.cuda import is_available
from clib.utils import Options


class TrainOptions(Options):
    def __init__(self):
        super().__init__("RCNN")
        self.update(
            {
                "device": "cuda" if is_available() else "cpu",
                "model_base_path": "path/to/folder/to/save/pth",
                "dataset_path": "path/to/dataset",
                "num_classes": 10,
                "use_relu": False,
                "use_max_pool": False,
                "train_mode": ["Holdout", "K-fold"][0],
                "seed": 42,
                "epochs": 2,
                "batch_size": 64,
                "lr": 0.001,
                "factor": 0.1,  # lr_this_turn  = lr_last_turn * factor
                "repeat": 2,  # number of turns
                "val": 0.2,  # when Holdout, present to validate (not for train!)
                "comment": "",
            }
        )

class TestOptions(Options):
    def __init__(self):
        super().__init__('RCNN')
        self.update(
            {
                'device': 'cuda' if is_available() else 'cpu',
                'model_path': 'path/to/model.pth',
                'dataset_path': 'path/to/dataset',
                'batch_size': 8,
                'num_classes': 10,
                'use_relu': False,
                'use_max_pool': False,
                'comment': ''
            }
        )