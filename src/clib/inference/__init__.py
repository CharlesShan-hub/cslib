import torch

class BaseInferencer: # 没想好怎么办，现在的 fusion 也能跑，先这样吧
    def __init__(self, opts, TestOptions, **kwargs):
        self.opts = TestOptions().parse(opts,present=False)
        self.set_seed()
    
    def inference(self):
        pass
    
    def set_seed(self):
        torch.manual_seed(self.opts.seed) # 设置随机种子（仅在CPU上）
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.opts.seed)
            torch.cuda.set_device(0)  # 假设使用第0号GPU