from clib.utils.gui import *

class UI(BasicUI):
    def __init__(self):
        super().__init__(
            title_label_text = 'Image Classification Demo',
            background = 'white',
            foreground = 'black'
        )
        self.title("Clib Pro")
    
    def config(self,**kwargs):
        super().config(**kwargs)
        self.configer_config = {
            'width': self.config_width,
            'label_width': self.config_label_width,
            'btn_width': self.config_label_width,
            'height': self.config_height,
            'background': self.background,
            'foreground': self.foreground,
        }
        
    def define(self):
        super().define()
        self.model_box = ConfigBox(
            master=self.config_frame,
            text='Model',
            values=['LeNet','AlexNet'],
            **self.configer_config
        )
        self.pth_path_btn = ConfigPath(
            master=self.config_frame,
            mode='file',
            text='Pre-Trained',
            **self.configer_config
        )
        self.dataset_box = ConfigBox(
            master=self.config_frame,
            text='Dataset',
            values=['MNIST'],
            **self.configer_config
        )
        self.dataset_path_btn = ConfigPath(
            master=self.config_frame,
            mode='dir',
            text='Dataset Path',
            **self.configer_config
        )
        self.load_btn = tk.Button(
            master=self.config_frame,
            text='load',
            background=self.background,
            foreground=self.foreground,
            command=lambda:self.load()
        )
        self.infer_btn = tk.Button(
            master=self.config_frame,
            text='infer',
            background=self.background,
            foreground=self.foreground,
            command=lambda:self.inference()
        )
        self.pics = [PicBox(master=self.show_frame,**self.configer_config) for _ in range(4)]
    
    def inference(self):
        pass

    def load(self):
        pass

    def pack(self):
        super().pack()
        self.model_box.pack()
        self.pth_path_btn.pack()
        self.dataset_box.pack()
        self.dataset_path_btn.pack()
        for i in self.pics:
            i.pack(side='left')
        self.load_btn.pack()
        self.infer_btn.pack()


    
'''

        



import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk

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

class App(tk.Tk):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.calculate()
        self.config()
        self.define()
        self.pack()
    
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
    
    def config(self):
        self.background = "black"
        self.foreground = "white"
        self.width = 100
        self.height = 360
        self.title_height = 3
        self.config_width = 15
        self.config_label_width = 10
        self.config_height = 3
        self.configer_config = {
            'width': self.config_width,
            'label_width': self.config_label_width,
            'height': self.config_height,
            'background': self.background,
            'foreground': self.foreground,
        }
        self.scale_factor = 6
    
    def define(self):
        self.title = tk.Label(
            master=self,
            text="Image Classify of MNist",
            height=self.title_height,
            width=self.width,
            background=self.background,
            foreground=self.foreground
        )
        self.content = tk.Frame(
            master=self,
            height=self.height - self.title_height,
            width=self.width,
            background=self.background
        )
        self.config_frame = tk.Frame(
            master=self.content,
            width=self.config_width,
            background=self.background
        )
        self.show_frame = tk.Frame(
            master=self.content,
            width=self.width - self.config_width,
            background=self.background
        )
        self.model_list = ['LeNet','AlexNet']
        self.model_name = ttk.Combobox(
            master = self.config_frame, 
            width = self.config_width,
            values = self.model_list,
            foreground=self.background
        )
        self.pre_trained_label = tk.Entry(
            master = self.config_frame,
            width=self.config_width,
            background=self.background,
            foreground=self.foreground
        )
        self.pre_trained = tk.Button(
            master = self.config_frame, 
            width = self.config_width,
            text = "Model Path",
            foreground=self.background,
            command=lambda:self.open_file(self.pre_trained_label)
        )
        self.dataset_list = ['MNist']
        self.dataset = ttk.Combobox(
            master = self.config_frame, 
            width = self.config_width,
            values = self.dataset_list,
            foreground=self.background
        )
        self.dataset_label = tk.Entry(
            master = self.config_frame,
            width=self.config_width,
            background=self.background,
            foreground=self.foreground
        )
        self.dataset_path = tk.Button(
            master = self.config_frame, 
            width = self.config_width,
            text = "Dataset Path",
            foreground=self.background,
            command=lambda:self.open_dir(self.dataset_label)
        )
        self.inference_button = tk.Button(
            master = self.config_frame, 
            width = self.config_width,
            text = "Inference",
            foreground=self.background,
            command=self.inference
        )
        self.pic_frame1 = tk.Frame(
            master=self.show_frame
        )
        self.pic1 = tk.Label(
            master=self.pic_frame1
        )
        self.pic2 = tk.Label(
            master=self.pic_frame1
        )
        self.pic_frame2 = tk.Frame(
            master=self.show_frame
        )
        self.pic3 = tk.Label(
            master=self.show_frame
        )
        self.pic4 = tk.Label(
            master=self.show_frame
        )
        self.pics = [ self.pic1,self.pic2,self.pic3,self.pic4 ]
        self.images = []
    
    def pack(self):
        # Title and Body
        self.title.pack(fill='x')
        self.content.pack(fill='both')
        self.config_frame.pack(fill='y',side='left')
        self.show_frame.pack(fill='both',side='left')
        # Config Panel
        self.model_name.pack(fill='x')
        self.pre_trained_label.pack(fill='x')
        self.pre_trained.pack(fill='x')
        self.dataset.pack(fill='x')
        self.dataset_label.pack(fill='x')
        self.dataset_path.pack(fill='x')
        self.inference_button.pack(fill='x')
        # Show Panel
        self.pic_frame1.pack(fill='x')
        self.pic_frame2.pack(fill='x')
        self.pic1.pack()
        self.pic2.pack()
        self.pic3.pack()
        self.pic4.pack()
    
    def open_file(self,file_label):
        file_path = filedialog.askopenfilename()
        if file_path:
            file_label.insert(0,file_path)
        else:
            file_label.insert(0,file_path)
    
    def open_dir(self,file_label):
        file_path = filedialog.askdirectory()
        if file_path:
            file_label.insert(0,file_path)
        else:
            file_label.insert(0,file_path)
    
    def inference(self):
        def _get_image(tensor):
            img = tensor_to_image(tensor)
            original_width, original_height = img.size
            new_width = int(original_width * self.scale_factor)
            new_height = int(original_height * self.scale_factor)
            return img.resize((new_width, new_height))

        with torch.no_grad():
            images,labels = next(self.dataLoader_iter)
            self.images = [ImageTk.PhotoImage(_get_image(i)) for i in images]
        

        # 放大图片
        
            for p,i in zip(self.pics,self.images):
                p.config(image = i)
            # print(f"label:{labels[0]}, inference:{torch.max(self.model(images[0]).data, 1)}")
            

if __name__ == "__main__":
    app = App()
    app.mainloop()
'''