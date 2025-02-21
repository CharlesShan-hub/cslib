from clib.metrics.utils import fusion_preprocessing
import torch
import torch.nn.functional as F

__all__ = [
    'con',
    'con_approach_loss',
    'con_metric'
]

# def con(A: torch.Tensor) -> torch.Tensor:
#     """
#     Calculate the Contrast (CON) metric of an image.
#     https://blog.csdn.net/zsc201825/article/details/89645190

#     Args:
#         A (torch.Tensor): The input image tensor.

#     Returns:
#         torch.Tensor: The calculated CON value.
#     """
#     # padding
#     A = torch.nn.functional.pad(A*255, (1, 1, 1, 1), mode='replicate')
#     # con
#     res1 = kornia.filters.filter2d(A,torch.tensor([[[0,1,0],[0,-1,0],[0,0,0]]]),padding='valid')
#     res2 = kornia.filters.filter2d(A,torch.tensor([[[0,0,0],[1,-1,0],[0,0,0]]]),padding='valid')
#     res3 = kornia.filters.filter2d(A,torch.tensor([[[0,0,0],[0,-1,1],[0,0,0]]]),padding='valid')
#     res4 = kornia.filters.filter2d(A,torch.tensor([[[0,0,0],[0,-1,0],[0,1,0]]]),padding='valid')

#     res1 = torch.sum(torch.abs(res1) ** 2)
#     res2 = torch.sum(torch.abs(res2) ** 2)
#     res3 = torch.sum(torch.abs(res3) ** 2)
#     res4 = torch.sum(torch.abs(res4) ** 2)

#     _,_,M,N = A.shape
#     M-=2
#     N-=2
#     return (res1+res2+res3+res4)/(4*M*N - 2*M - 2*N)

def con(A: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Contrast (CON) metric of an image.

    Args:
        A (torch.Tensor): The input image tensor (C, H, W) or (B, C, H, W).

    Returns:
        torch.Tensor: The calculated CON value.
    """
    # Ensure input is in the range [0, 255]
    A = A * 255

    # Ensure input is a 4D tensor (B, C, H, W)
    if A.dim() == 3:
        A = A.unsqueeze(0)  # Add batch dimension if missing
    if A.size(1) != 1:
        raise ValueError("Input tensor must be a grayscale image with 1 channel.")

    # Define the Sobel-like kernels for computing differences in four directions
    kernels = torch.tensor([
        [[0, 1, 0], [0, -1, 0], [0, 0, 0]],  # Horizontal
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]],  # Vertical
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],  # Diagonal (top-left to bottom-right)
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]]   # Anti-diagonal (top-right to bottom-left)
    ], dtype=A.dtype, device=A.device).unsqueeze(1)  # Shape: (4, 1, 3, 3)

    # Apply padding and convolution
    A_padded = F.pad(A, (1, 1, 1, 1), mode='replicate')  # Pad by 1 pixel on each side
    responses = F.conv2d(A_padded, kernels, padding=0)  # Shape: (B, 4, H-2, W-2)

    # Compute the sum of squared differences
    squared_responses = torch.sum(torch.abs(responses) ** 2, dim=(1, 2, 3))  # Sum over spatial dimensions
    total_sum = torch.sum(squared_responses)  # Sum over all directions

    # Calculate the normalization factor
    _, _, H, W = A.shape
    norm_factor = 4 * H * W - 2 * H - 2 * W

    # Return the normalized contrast value
    return total_sum / norm_factor

def con_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(con(A)-con(F))

@fusion_preprocessing
def con_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return con(F)

def main():
    from clib.metrics.fusion import ir, vis, fused

    toy = torch.tensor([[[[1,3,9,9],[2,1,3,7],[3,6,0,6],[6,8,2,0]]]])/255.0

    print(f'CON(toy):{con_metric(toy,toy,toy)}') # should be 13.333
    print(f'CON(ir):{con_metric(ir,ir,ir)}')
    print(f'CON(vis):{con_metric(vis,vis,vis)}')
    print(f'CON(fused):{con_metric(fused,fused,fused)}')

if __name__ == '__main__':
    main()
