from torch.cuda import is_available
from ....utils import Options


class TrainOptions(Options):
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                         Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --pre_trained       Str            model.pth                     The path of pre-trained model
        ----------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        super().__init__('LeNet')
        self.update({
            'device': 'cuda' if is_available() else 'cpu',
            'dataset_path': '../../data/mnist', 
            'epochs': 1, 
            'batch_size': 64, 
            'lr': 0.001, 
        })


class TestOptions(Options):
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                         Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --pre_trained       Str            model.pth                     The path of pre-trained model
        ----------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        super().__init__('LeNet')
        self.update({
            'pre_trained': 'model.pth',
            'device': 'cuda' if is_available() else 'cpu',
        })
