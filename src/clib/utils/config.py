"""
Config Utils Module

Contains Base Options class.
"""

from typing import Dict, Any
from argparse import Namespace
from pathlib import Path
from os import makedirs
import json

__all__ = [
    'Options'
]

class Options(Namespace):
    """
    Base Options class.

    This class provides a way to define and update command line arguments.

    Attributes:
        opts (argparse.Namespace): A namespace containing the parsed command line arguments.

    Methods:
        INFO(string): Print an information message.
        presentParameters(args_dict): Print the parameters setting line by line.
        update(parmas): Update the command line arguments.
    
    Example: 
        * config.py in a specific algorithm
        >>> from torch.cuda import is_available
        >>> from xxx import Options
        >>> class TestOptions(Options):
        >>> def __init__(self):
        >>>     super().__init__('DenseFuse')
        >>>     self.update({
        >>>         'pre_trained': 'model.pth',
        >>>         'device': 'cuda' if is_available() else 'cpu'
        >>>     })

        * update TestOptions in other files
        >>> opts = TestOptions().parse(other_opts_dict)

        * use TestOptions in other files
        >>> pre_trained_path = opts.pre_trained
    """

    def __init__(self, name: str = 'Undefined', params: Dict[str, Any] = {}):
        # self.opts = Namespace()
        self.name = name
        if len(params) > 0:
            self.update(params)

    def INFO(self, string: str):
        """
        Print an information message.

        Args:
            string (str): The message to be printed.
        """
        print("[ %s ] %s" % (self.name,string))

    def presentParameters(self):
        """
        Print the parameters setting line by line.

        Args:
            args_dict (Dict[str, Any]): A dictionary containing the command line arguments.
        """
        self.INFO("========== Parameters ==========")
        for key in sorted(vars(self).keys()):
            self.INFO("{:>15} : {}".format(key, getattr(self, key)))
        self.INFO("===============================")

    def update(self, parmas: Dict[str, Any] = {}):
        """
        Update the command line arguments.

        Args:
            parmas (Dict[str, Any]): A dictionary containing the updated command line arguments.
        """
        for (key, value) in parmas.items():
            setattr(self, key, value)
    
    def parse(self, parmas: Dict[str, Any] = {}, present: bool = True):
        """
        Update the command line arguments. Can also present into command line.
        
        Args:
            parmas (Dict[str, Any]): A dictionary containing the updated command line arguments.
            present (bool) = True: Present into command line.
        """
        self.update(parmas)
        if present:
            self.presentParameters()
        return self
    
    def save(self, models_path: str = ''):
        """
        Save Config when train is over.

        Args:
            params
        """
        if models_path == '':
            models_path = self.models_path
        models_path = Path(models_path) # type: ignore
        if models_path.exists() == False: # type: ignore
            makedirs(models_path)
        with open(Path(models_path,'config.txt'), 'w') as f:
            for key in sorted(vars(self).keys()):
                f.write("{:>15} : {}\n".format(key, getattr(self, key)))
        # print(self)
    
    # def __str__(self):
    #     print({key: getattr(self, key) for key in vars(self).keys()})
    #     return json.dumps({key: getattr(self, key) for key in vars(self).keys()},indent=4)
        