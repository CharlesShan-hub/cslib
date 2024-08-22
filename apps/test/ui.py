import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk

# class Config(tk.Frame):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         self.config(**kwargs)
#         self.define()
#         self.pack()
    
#     def config(self,**kwargs):
#         self.label_width = kwargs['label_width']
#         self.width = kwargs['width']

#     def define(self):
#         self.label = tk.Label(
#             master = self,
#             width = self.label_width,
#         )

#     def pack(self):
#         pass

class App(tk.Tk):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.config()
        self.define()
        self.pack()
    
    def config(self):
        self.background = "black"
        self.foreground = "white"
        self.width = 80
        self.height = 360
        self.title_height = 3
        self.config_width = 15
        self.config_height = 3
    
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
        self.model = ttk.Combobox(
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
    
    def pack(self):
        # Title and Body
        self.title.pack(fill='x')
        self.content.pack(fill='both')
        self.config_frame.pack(fill='y')
        self.show_frame.pack(fill='both')
        # Config Panel
        self.model.pack(fill='x')
        self.pre_trained_label.pack(fill='x')
        self.pre_trained.pack(fill='x')
        self.dataset.pack(fill='x')
        self.dataset_label.pack(fill='x')
        self.dataset_path.pack(fill='x')
        self.inference.pack(fill='x')
        # Show Panel
        四张图片，每张图片下边是一个 label
    
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
    
    def inference(self,)

if __name__ == "__main__":
    app = App()
    app.mainloop()
