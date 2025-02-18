import torch

# 假设我们有一个 BCHW 形状的图片张量，其中 B=1, C=3, H=2, W=2
# 创建一个 BCHW 形状的张量
image_tensor = torch.tensor([[[[1, 2], [3, 4]],
                              [[5, 6], [7, 8]],
                              [[9, 10], [11, 12]]]], dtype=torch.float32)

# 创建一个 3x3 的二维张量
matrix = torch.tensor([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=torch.float32)

# 对每个 HxW 的块应用矩阵乘法
# 首先将 image_tensor 转换为 BH'W'xC 的形状，其中 H' = H, W' = W
# 然后使用广播机制进行矩阵乘法
result = torch.einsum('bchw,cd->bhdw', image_tensor, matrix)
print(result)
breakpoint()