from ui import UI

from clib.utils.io import tensor_to_image
from clib.model import classify 
from clib.metrics.fusion import ir 
import sys
from pathlib import Path
sys.path.append(Path(__file__,'../../../scripts').resolve().__str__())
import config

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageTk # type: ignore
import numpy as np
import matplotlib.pyplot as plt

class App(UI):
    def __init__(self):
        super().__init__()
        self.calculate()
        self.infer_btn.config(command=lambda:self.inference())

    def calculate(self):
        model_name = 'LeNet'
        config.opts[model_name] = {
            '*ResPath': r'@ModelBasePath/LeNet/MNIST/',
            '*pre_trained': r'@ResPath/9775/model.pth'
        }
        opts = config.opts[model_name]
        self.alg = getattr(classify,model_name)
        self.opts = self.alg.TestOptions().parse(opts,present=False)
        self.model = self.alg.load_model(self.opts)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(root=self.opts.TorchVisionPath, train=False, download=True, transform=transform)
        self.dataLoader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)
        self.dataLoader_iter = iter(self.dataLoader)

        self.opts.presentParameters()
        self.model.eval()
    
    def inference(self):
        pass
    #     def _get_image(tensor):
    #         img = tensor_to_image(tensor)
    #         original_width, original_height = img.size
    #         new_width = int(original_width * self.scale_factor)
    #         new_height = int(original_height * self.scale_factor)
    #         return img.resize((new_width, new_height))

    #     with torch.no_grad():
    #         images,labels = next(self.dataLoader_iter)
    #         self.images = [ImageTk.PhotoImage(_get_image(i)) for i in images]
    #         # 放大图片
    #         for p,i in zip(self.pics,self.images):
    #             p.config(image = i)

if __name__ == "__main__":
    app = App()
    app.mainloop()
