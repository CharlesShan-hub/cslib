from clib.metrics.utils import fusion_preprocessing
import torch

__all__ = [
    'mb',
    'mb_approach_loss',
    'mb_metric'
]

def mb(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean bias (MB).

    Args:
        I (torch.Tensor): Input tensor A.
        B (torch.Tensor): Input tensor B.
        F (torch.Tensor): Input tensor F.

    Returns:
        torch.Tensor: The edge intensity of the input tensors.
    """

    [mA, mB, mF] = [torch.mean(I) for I in [A, B, F]]

    return torch.abs(1-2*mF/(mA+mB)) # 我加的绝对值

# 如果两幅图相等，MB 会一致
def mb_approach_loss(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return mb(A, B, F)

@fusion_preprocessing
def mb_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return mb(A, B, F) # 不乘 255 也一样

if __name__ == '__main__':
    from clib.metrics.fusion import vis,ir,fused
    print(mb_metric(ir,vis,fused).item())
