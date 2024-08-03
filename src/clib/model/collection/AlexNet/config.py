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
        super().__init__('AlexNet')
        self.update({
            'pre_trained': 'model.pth',
            'device': 'cuda' if is_available() else 'cpu',
            'dataset_path': '../../data/mnist', 
            'epochs': 200, 
            'batch_size': 64, 
            'lr': 0.0002, 
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
        super().__init__('AlexNet')
        self.update({
            'pre_trained': 'model.pth',
            'device': 'cuda' if is_available() else 'cpu',
        })
