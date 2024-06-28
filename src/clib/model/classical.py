import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 网络层(原始的LeNet-5使用平均池化层而不是最大池化层)
        self.convnet = nn.Sequential(
            # 第一层卷积，6个5x5的卷积核，输入通道为1（灰度图），输出通道为6
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 第二层卷积，16个5x5的卷积核，输入通道为6，输出通道为16
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            # 全连接层，120个节点
            nn.Linear(256, 120),
            nn.Tanh(),
            # 全连接层，84个节点
            nn.Linear(120, 84),
            nn.Tanh(),
            # 输出层，使用softmax激活函数进行分类
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # 批量数自适应得到，通道数为1，图片为28X28
        x = x.view(-1,1,28,28)
        # 通过卷积层
        x = self.convnet(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fc(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1) 
        x = self.classifier(x)
        return x