import torch
import torch.nn as nn

def load_model(opts):
    model = LeNet().to(opts.device)
    params = torch.load(opts.pre_trained, map_location=opts.device)
    model.load_state_dict(params)
    return model

class LeNet(nn.Module):
    def __init__(self, num_classes=10, use_relu=False, use_max_pool=False):
        super(LeNet, self).__init__()
        # 网络层(原始的LeNet-5使用平均池化层而不是最大池化层)
        self.convnet = nn.Sequential(
            # 第一层卷积，6个5x5的卷积核，输入通道为1（灰度图），输出通道为6
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU() if use_relu else nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2) if use_max_pool else nn.AvgPool2d(kernel_size=2, stride=2),
            # 第二层卷积，16个5x5的卷积核，输入通道为6，输出通道为16
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU() if use_relu else nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2) if use_max_pool else nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            # 全连接层，120个节点
            nn.Linear(256, 120),
            nn.ReLU() if use_relu else nn.Tanh(),
            # 全连接层，84个节点
            nn.Linear(120, 84),
            nn.ReLU() if use_relu else nn.Tanh(),
            # 输出层，使用softmax激活函数进行分类
            nn.Linear(84, num_classes)
        )

        # 初始化权重和偏差
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if m.bias is not None else 'leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 批量数自适应得到，通道数为1，图片为28X28
        x = x.view(-1,1,28,28)
        # 通过卷积层
        x = self.convnet(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 通过全连接层
        return self.fc(x)