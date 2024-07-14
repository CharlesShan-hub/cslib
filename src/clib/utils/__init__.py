from typing import Dict,Any
import argparse
import matplotlib.pyplot as plt
import numpy as np

def path_load_test():
    print("Hello World!")

def glance(tensor):
    # 处理色彩
    if len(tensor.shape)==3:
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        image = tensor.detach().cpu().numpy()

    # 使用 PIL 显示图片
    if len(image.shape) == 2:
        plt.imshow((image * 255).astype(np.uint8),cmap='gray')
    else:
        plt.imshow((image * 255).astype(np.uint8))
    plt.show()

class Options(object):
    """
    Options class for DeepFuse.

    This class provides a way to define and update command line arguments.

    Attributes:
        opts (argparse.Namespace): A namespace containing the parsed command line arguments.

    Methods:
        INFO(string): Print an information message.
        presentParameters(args_dict): Print the parameters setting line by line.
        update(parmas): Update the command line arguments.
    """

    def __init__(self, params: Dict[str, Any] = {}):
        self.opts = argparse.Namespace()
        if len(params) > 0:
            self.update(params)

    def INFO(self, string: str):
        """
        Print an information message.

        Args:
            string (str): The message to be printed.
        """
        print("[ DeepFuse ] %s" % (string))

    def presentParameters(self, args_dict: Dict[str, Any]):
        """
        Print the parameters setting line by line.

        Args:
            args_dict (Dict[str, Any]): A dictionary containing the command line arguments.
        """
        self.INFO("========== Parameters ==========")
        for key in sorted(args_dict.keys()):
            self.INFO("{:>15} : {}".format(key, args_dict[key]))
        self.INFO("===============================")

    def update(self, parmas: Dict[str, Any]):
        """
        Update the command line arguments.

        Args:
            parmas (Dict[str, Any]): A dictionary containing the updated command line arguments.
        """
        for (key, value) in parmas.items():
            setattr(self.opts, key, value)
