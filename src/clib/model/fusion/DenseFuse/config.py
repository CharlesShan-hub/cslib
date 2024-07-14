import argparse
from torch.cuda import is_available
from pathlib import Path
from ....utils import Options

class TestOptions(Options):
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                         Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --pre_trained       Str         model.pth                       The path of pre-trained model
            --H                 Int         400                             The height of the result image
            --W                 Int         600                             The width of the result image
        ----------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        super().__init__('DenseFuse')
        parser = argparse.ArgumentParser()
        # parser.add_argument('--H'          , type = int, default = 400)
        # parser.add_argument('--W'          , type = int, default = 600)
        parser.add_argument('--strategy_type', type = str, default = 'addition')# attention_weight
        self.opts = parser.parse_args()
        self.opts.device = 'cuda' if is_available() else 'cpu'

    def parse(self,parmas={}):
        self.update(parmas)
        self.presentParameters(vars(self.opts))
        return self.opts