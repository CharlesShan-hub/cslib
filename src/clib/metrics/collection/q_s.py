from clib.metrics.utils import fusion_preprocessing
import torch
import kornia

__all__ = [
    'q_s',
    'q_s_approach_loss',
    'q_s_metric'
]

def q_s(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor,
        window_size: int = 11, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the Peilla's quality index (q_s) for image fusion.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        window_size (int, optional): The size of the Gaussian kernel for filtering. Default is 11.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The q_w quality index between the two input images and their fusion.
    """

    def sigma2(A):
        kernel = kornia.filters.get_gaussian_kernel1d(window_size, 1.5, device=A.device, dtype=A.dtype)
        mu = kornia.filters.filter2d_separable(A, kernel, kernel, padding="valid")
        mu2 = kornia.filters.filter2d_separable(A**2, kernel, kernel, padding="valid")
        return mu2 - mu**2

    def _ssim(A,B):
        C1 = (0.01*255)**2
        C2 = (0.03*255)**2
        kernel = kornia.filters.get_gaussian_kernel1d(window_size, 1.5, device=A.device, dtype=A.dtype)
        muA = kornia.filters.filter2d_separable(A, kernel, kernel, padding="valid")
        muB = kornia.filters.filter2d_separable(B, kernel, kernel, padding="valid")
        sAA = kornia.filters.filter2d_separable(A**2, kernel, kernel, padding="valid") - muA**2
        sBB = kornia.filters.filter2d_separable(B**2, kernel, kernel, padding="valid") - muB**2
        sAB = kornia.filters.filter2d_separable(A*B, kernel, kernel, padding="valid") - muA*muB

        return  ((2*muA*muB + C1)*(2*sAB + C2)) / ((muA**2 + muB**2 + C1)*(sAA + sBB + C2));

    sigma2A_sq = sigma2(A*255)
    sigma2B_sq = sigma2(B*255)

    rectify = ((sigma2A_sq + sigma2B_sq) < eps).float() * 0.5
    sigma2A_sq = sigma2A_sq + rectify
    sigma2B_sq = sigma2B_sq + rectify
    ramda = sigma2A_sq / (sigma2A_sq + sigma2B_sq)

    ssimAF = _ssim(A, F)
    ssimBF = _ssim(B, F)

    return torch.mean(ramda * ssimAF + (1-ramda) * ssimBF)

def q_s_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1-q_s(A, A, F, window_size=11, eps=1e-10)

@fusion_preprocessing
def q_s_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return q_s(A*255.0, B*255.0, F*255.0, window_size=11, eps=1e-10)

if __name__ == '__main__':
    from clib.metrics.fusion import vis,ir,fused
    print(q_s_metric(ir,vis,fused).item())
