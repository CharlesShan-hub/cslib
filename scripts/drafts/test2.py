import torch
import torch.nn.functional as F

# 假设我们有一个 BCHW 形状的图片张量，这里随机生成一个示例
# 假设批量大小 B=1，通道数 C=3，高度 H=5，宽度 W=5
B, C, H, W = 1, 3, 5, 5
image_tensor = torch.randn(B, C, H, W)

# 定义一个3x3的二维张量，这里随机生成一个示例
# 这将作为卷积核
kernel = torch.randn(3, 3)

# 将3x3的二维张量扩展为4D张量，使其能够进行卷积操作
# 扩展后的形状为 [out_channels, in_channels, kernel_height, kernel_width]
# 由于我们是对每个通道进行卷积，所以out_channels和in_channels都设置为1
kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)

# 进行卷积操作
# stride=1, padding=0 是默认参数，可以根据需要进行调整
convolved = F.conv2d(image_tensor, kernel_expanded, stride=1, padding=0)

print("Image Tensor:\n", image_tensor)
print("Kernel:\n", kernel)
print("Convolved Tensor:\n", convolved)
