"""
Config Utils Module

Contains Base Options class for clib module and user module.
"""

from typing import Dict, List, Any
from argparse import Namespace
from pathlib import Path
from os import makedirs
import json
from collections import UserDict
import torch

__all__ = [
    'Options',
    'ConfigDict',
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
    
    def save(self, src: str = ''):
        """
        Save Config when train is over.

        Args:
            params
        """
        src = self.ResBasePath if src == '' else src
        _src: Path = Path(src) # type: ignore
        if _src.exists() == False:
            makedirs(_src)
        with open(Path(_src,'config.json'), 'w') as f:
            f.write(self.__str__())
    
    def __str__(self):
        return json.dumps({
            key: getattr(self, key).__str__() for key in vars(self).keys()
        },indent=4)
        

class ConfigDict(UserDict):
    """
    ConfigDict is declared when calling clib. It is used to specify the highest 
    priority parameters. It requires a list of data root directories, and the 
    dictionary will automatically find the first existing root directory, which 
    is used to adapt to multi-platform scenarios.
    
    The directory structure of your data root directory needs to be as follows:
    - data_root_path
      | - torchvision: Contains the official datasets of torchvision
      | - model: Contains the training results and pre-trained models of each model
      | - Fusion: Contains data for image fusion
      | - SR: Contains data for super-resolution reconstruction
      | - ...: Contains other data, which users can define and extend as needed
    
    Example: # In config.py
    >>> from pathlib import Path
    >>> from clib.utils import ConfigDict
    >>> opts = ConfigDict([
                    '/root/autodl-fs/DateSets',
                    '/Volumes/Charles/DateSets',
                    '/Users/kimshan/resources/DataSets'
                ])
    >>> opts['LeNet'] = {
            'ResBasePath': Path(opts.ModelBasePath,'LeNet','MNIST','temp'),
            'pre_trained': Path(opts.ModelBasePath,'LeNet','MNIST','9839_m1_d003','model.pth'),
        }
    """
    def __init__(self, data_root_path_list: List[str]):
        super().__init__({})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root_path_list = data_root_path_list
        self.DataRootPath = None
        for path in self.data_root_path_list:
            if Path(path).exists() and Path(path).is_dir():
                self.DataRootPath = path
                break
        assert(self.DataRootPath is not None)
        self.TorchVisionPath = Path(self.DataRootPath, "torchvision").__str__()
        self.FusionPath = Path(self.DataRootPath, "Fusion").__str__()
        self.SRPath = Path(self.DataRootPath, "SR").__str__()
        self.ModelBasePath = Path(self.DataRootPath, "Model").__str__()

    def __setitem__(self, key: str, value: Dict[str, Any]):
        check_list = [
            'device','DataRootPath','TorchVisionPath',
            'FusionPath','SRPath','ModelBasePath',
        ]
        for item in check_list:
            if item not in value:
                value[item] = getattr(self,item)
        
        for item in list(value.keys()):
            if item.startswith('*'):
                temps = value[item] if isinstance(value[item],list) else [value[item]]
                for i,temp in enumerate(temps):
                    part_list = []
                    for part in Path(temp).parts:
                        if not part.startswith('@'):
                            part_list.append(part)
                        else:
                            if hasattr(self,part[1:]):
                                part_list.append(getattr(self,part[1:]))
                            else:
                                part_list.append(value[part[1:]])
                    temps[i] = Path(*part_list).__str__()
                value[item[1:]] = temps[0] if len(temps)==1 else temps
                
        super().__setitem__(key, {k: value[k] for k in value if not k.startswith('*')})