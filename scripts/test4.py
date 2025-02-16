import torch
import torch.nn.functional as F

def split_img(X, n=3):
    def mirror_pad_if_odd_and_unfold(tensor):
        # padding
        pad_height = 0 if (tensor.shape[-2] % 2 == 0) else 1
        pad_width = 0 if (tensor.shape[-1] % 2 == 0) else 1
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode='reflect')
        
        # unfold
        H, W = tensor.shape[-2]//2, tensor.shape[-1]//2
        return torch.cat([
            tensor[..., :H, :W], # Top - Left
            tensor[..., :H,-W:], # Top - Right
            tensor[...,-H:, :W], # Bottom - Left
            tensor[...,-H:,-W:], # Bottom - Right
        ],dim=-3), (pad_height,pad_width)
    
    paddings = []
    for i in range(n):
        X, pad = mirror_pad_if_odd_and_unfold(X)
        paddings.append(pad)
    return X,paddings

def merge_img(X, paddings, n=3):
    def fold(tensor,pad):
        (B,C,H,W) = tensor.shape
        C//=4
        padded = torch.zeros(size=(B,C,H*2,W*2))
        padded[:,:,:H,:W] = tensor[:,0*C:1*C,:,:]
        padded[:,:,:H,-W:] = tensor[:,1*C:2*C,:,:]
        padded[:,:,-H:,:W] = tensor[:,2*C:3*C,:,:]
        padded[:,:,-H:,-W:] = tensor[:,3*C:4*C,:,:]
        return padded[:,:,:H*2-pad[0],:W*2-pad[1]]

    for i in range(n):
        X = fold(X, paddings.pop())
    return X

# 示例使用
# 创建一个示例张量，形状为 (C, H, W)
a1 = torch.rand(1, 1, 16, 16)
a2 = torch.rand(2, 1, 16, 15)
a3 = torch.rand(1, 3, 15, 14)
a4 = torch.rand(8, 3, 16, 17)

b1,p1 = split_img(a1)
b2,p2 = split_img(a2)
b3,p3 = split_img(a3)
b4,p4 = split_img(a4)

c1 = merge_img(b1,p1)
c2 = merge_img(b2,p2)
c3 = merge_img(b3,p3)
c4 = merge_img(b4,p4)

print(a1.equal(c1))
print(a2.equal(c2))
print(a3.equal(c3))
print(a4.equal(c4))

# breakpoint()