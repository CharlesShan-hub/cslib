from typing import List, Union, Optional
from typing_extensions import override
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

__all__ = [
    'Laplacian',
    'Contrust',
    'FSD',
    'Graident',
    'Morphological',
]

class Base(object):
    """
    Represents a pyramid constructed from an input image using Gaussian pyramid(Default).

    Attributes:
        name (str): Name of the pyramid.
        image (Union[str, torch.Tensor]): Input image as a filename (str) or tensor (torch.Tensor).
        pyramid (List[torch.Tensor]): List to store pyramid layers.
        layer (int): Number of layers in the pyramid.
        recon (torch.Tensor): Output image after reconstruction.
        auto (bool): Flag indicating whether to automatically construct and reconstruct the pyramid.
        down_way (str): Method used for downsampling during pyramid construction.
        up_way (str): Method used for upsampling during pyramid reconstruction.
        dec_way (str): Method used for decomposition.
        rec_way (str): Method used for reconstruction.
    """
    def __init__(self, name: str, **kwargs) -> None:
        """
        Initializes a Pyramid object.

        Args:
            name (str): Name of the pyramid.
            **kwargs: Additional keyword arguments to customize object attributes.

        Attributes:
            name (str): Name of the pyramid.
            image (Union[str, torch.Tensor]): Input image as a filename (str) or tensor (torch.Tensor).
            pyramid (List[torch.Tensor]): List to store pyramid layers.
            layer (int): Number of layers in the pyramid.
            recon (torch.Tensor): Output image after reconstruction.
            auto (bool): Flag indicating whether to automatically construct and reconstruct the pyramid.
            down_way (str): Method used for downsampling during pyramid construction.
            up_way (str): Method used for upsampling during pyramid reconstruction.
            dec_way (str): Method used for decomposition.
            rec_way (str): Method used for reconstruction.
        """
        self.name = name
        self.image = None # type: ignore
        self.pyramid = []
        self.layer = 5
        self.recon = None
        self.auto = True
        self.down_way = 'zero'     # Downsample method
        self.up_way = 'zero'       # Upsample method
        self.dec_way = 'ordinary'  # Decomposition method
        self.rec_way = 'ordinary'  # Reconstruction method

        # Update attributes based on additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Default operations
        if self.image is not None:
           self.set_image(self.image)
        elif self.pyramid != []:
            # Auto Reconstruction for Output Image
            if self.auto:
                self.reconstruction()

    @staticmethod
    def gaussian_blur(image: torch.Tensor, kernel_size: int = 5,
        gau_blur_way: str = 'Pytorch', sigma: Optional[List[float]] = None, bias: float = 0) -> torch.Tensor:
        """
        Applies Gaussian blur to the input image.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            kernel_size (int, optional): Size of the Gaussian kernel. Default is 5.
            gau_blur_way (str, optional): Method used for Gaussian blur. Options are 'Pytorch' and 'Paper'. Default is 'Paper'.
            sigma (List[float], optional): Standard deviation for the Gaussian kernel. Only used when `gau_blur_way` is 'Pytorch'. Default is None.
            bias (float, optional): Bias value added to the Gaussian kernel. Default is 0.

        Returns:
            torch.Tensor: Blurred image tensor.

        Raises:
            ValueError: If `kernel_size` is not 3 or 5 when `gau_blur_way` is 'Paper', or if `gau_blur_way` is not 'Pytorch' or 'Paper'.
        """
        if gau_blur_way == "Pytorch":
            return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=sigma)
        elif gau_blur_way == "Paper":
            # Define a Gaussian kernel
            if kernel_size == 3:
                kernel = torch.tensor([[1., 2., 1.],
                                       [2., 4., 2.],
                                       [1., 2., 1.]]) / 16 + bias
            elif kernel_size == 5:
                kernel = torch.tensor([[1., 4., 6., 4., 1.],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]]) / 256 + bias
            else:
                raise ValueError(f"kernel size in paper only be 3 or 5, not {kernel_size}")
            # Expand dimensions of the kernel for convolution
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(image.dtype)

            # Adjust the kernel to match the number of channels in the input image
            kernel = kernel.expand(image.shape[1], -1, -1, -1)

            # Apply 2D convolution with the Gaussian kernel
            return F.conv2d(image, kernel, stride=1, padding=(kernel_size - 1) // 2, groups=image.shape[1])
        else:
            raise ValueError(f"`gau_blur_way` should only be 'Pytorch' or 'Paper', not {gau_blur_way}.")

    @staticmethod
    def down_sample(image: torch.Tensor, dawn_sample_way: str = "Zero") -> torch.Tensor:
        """
        Downsamples the input image using specified method.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            down_sample_way (str, optional): The method used for downsampling. Options are 'Max' for max pooling
                or 'Zero' for simple element removal. Defaults to 'Zero'.

        Returns:
            torch.Tensor: Downsampled image tensor.

        Raises:
            ValueError: If `down_sample_way` is not 'Max' or 'Zero'.
        """
        if dawn_sample_way == "Max":
            # Method 1. Subsample the image using 2x2 max pooling
            return F.max_pool2d(image, kernel_size=2, stride=2)
        elif dawn_sample_way == "Zero":
            # Method 2. Downsamples the input image by remove elements.
            return image[:, :, ::2, ::2]
        else:
            raise ValueError("`dawn_sample_way` should be 'Max' or 'Zero'")

    @staticmethod
    def up_sample(image: torch.Tensor) -> torch.Tensor:
        """
        Upsamples the input image using zero-padding.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Upsampled image tensor.
        """
        # Perform zero-padding
        batch_size, channels, height, width = image.size()
        padded_img = torch.zeros(batch_size, channels, 2 * height, 2 * width, device=image.device)
        padded_img[:, :, ::2, ::2] = image
        return padded_img

    @staticmethod
    def pyr_down(image: torch.Tensor) -> torch.Tensor:
        """
        Downsamples the input image using Gaussian blur and max pooling.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Downsampled image tensor.
        """
        blurred = Base.gaussian_blur(image)
        downsampled = Base.down_sample(blurred)
        return downsampled

    @staticmethod
    def pyr_up(image: torch.Tensor) -> torch.Tensor:
        """
        Upsamples the input image using zero-padding and Gaussian blur.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Upsampled image tensor.
        """
        padded_img = Base.up_sample(image)
        blurred_img = Base.gaussian_blur(padded_img)
        return blurred_img * 4

    def _build_gaussian_pyramid(self) -> None:
        """
        Constructs a Gaussian pyramid from the input image.
        """
        if self.image is None:
            raise ValueError("You should first assign a image.")
        image = self.image
        _, _, width, height = image.shape
        if self.layer > int(torch.floor(torch.log2(torch.tensor(min(width, height)))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.layer))
        self.gaussian = [image]
        for _ in range(self.layer):
            image = self.pyr_down(image)
            self.gaussian.append(image)

    def _build_base_pyramid(self) -> None:
        """
        Builds the base (default Gaussian) pyramid from the input image.
        """
        self._build_gaussian_pyramid()

    def _init_after_change_image(self) -> None:
        """
        Initializes after set a image.
        """
        self._build_base_pyramid()

    def set_image(self, image: Union[torch.Tensor, str], auto: Optional[bool] = None) -> None:
        """
        Sets the input image for the pyramid and performs automatic decomposition and reconstruction if specified.

        Args:
            image (Union[torch.Tensor, str]): Input image as a tensor or filename.
            auto (Optional[bool]): Flag indicating whether to automatically decompose and reconstruct the pyramid. Default is None.
        """
        # Converts input image filename to tensor if necessary.
        if isinstance(image, str):
            transform = transforms.Compose([transforms.ToTensor()])
            self.image: torch.Tensor = TF.to_tensor(Image.open(image)).unsqueeze(0)

        assert self.image.dim() == 4, 'Image batch must be of shape [N,C,H,W]'

        # Build Base(Defaule Gaussian) Pyramid from input image
        self._init_after_change_image()

        # Auto Decomposition to Pyramid and Reconstruction for Output Image
        if auto is not None:
            self.auto = auto
        if self.auto:
            self.decomposition()
            self.reconstruction()

    def decomposition(self, method: Optional[str] = None) -> None:
        """
        Decomposes the image into pyramid layers based on the specified method.

        Args:
            method (str, optional): Method used for decomposition. Defaults to None.
        """
        if self.image is None:
            raise ValueError("No image to do decomposition!")
        if method is not None:
            self.dec_way = method
        if self.dec_way is not None:
            decomposition_method = getattr(self, f"decomposition_{self.dec_way}", None)
            if decomposition_method is not None and callable(decomposition_method):
                decomposition_method()
            else:
                raise ValueError(f"Invalid decomposition method (reconstruct_{self.dec_way}):", method)
        else:
            raise ValueError("No decomposition method specified")

    def reconstruction(self, method: Optional[str] = None) -> None:
        """
        Reconstructs the image from the pyramid layers based on the specified method.

        Args:
            method (str, optional): Method used for reconstruction. Defaults to None.
        """
        if method is not None:
            self.rec_way = method
        if self.rec_way is not None:
            reconstruct_method = getattr(self, f"reconstruction_{self.rec_way}", None)
            if reconstruct_method is not None and callable(reconstruct_method):
                reconstruct_method()
            else:
                raise ValueError(f"Invalid reconstruct method (reconstruct_{self.rec_way}):", method)
        else:
            raise ValueError("No reconstruct method specified")

    def __getitem__(self, index: Union[int, slice]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get a layer or a subset of layers from the pyramid.

        Args:
            index (Union[int, slice]): Index or slice to select the layers.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Selected layer(s) from the pyramid.
        """
        return self.pyramid[index]

    def __setitem__(self, index: int, value: torch.Tensor) -> None:
        """
        Set a layer in the pyramid.

        Args:
            index (int): Index of the layer to be set.
            value (torch.Tensor): Tensor value to set the layer.
        """
        self.pyramid[index] = value

    def __len__(self) -> int:
        """
        Get the number of layers in the pyramid.

        Returns:
            int: Number of layers in the pyramid.
        """
        return len(self.pyramid)

    def append(self, item: torch.Tensor) -> None:
        """
        Appends a layer to the pyramid.

        Args:
            item (torch.Tensor): Pyramid layer tensor to append.
        """
        self.pyramid.append(item)


class Demo(Base):
    """
    Represents a demo class for showcasing pyramid development.

    This class inherits from the Base class and provides methods for decomposition and reconstruction
    of the pyramid using ordinary techniques.

    Attributes:
        Inherits attributes from the Base class.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a Demo object.

        Args:
            **kwargs: Additional keyword arguments to customize object attributes.

        Inherits attributes from the Base class.
        """
        # Add your own default params
        # self.layer = 5
        # self.dec_way = 'ordinary'  # Decomposition method
        # self.rec_way = 'ordinary'  # Reconstruction method

        # Do base default params
        super().__init__("Demo", **kwargs)

    # def _build_user_designed_base_pyramid(self) -> None:
    #     """
    #     Constructs your own pyramid from the input image.
    #     """
    #     if self.image is not None:
    #         pass
    #         # your own code
    #         #
    #         # Example: gaussian
    #         # image = self.image
    #         # self.gaussian = [image]
    #         # for _ in range(self.layer):
    #         #     image = self.pyr_down(image)
    #         #     self.gaussian.append(image)
    #     else:
    #         raise ValueError("You should first assign a image.")

    # @override
    # def _build_base_pyramid(self) -> None:
    #     """
    #     Builds the base (default Gaussian) pyramid from the input image.
    #     """
    #     # This is the default gaussian option
    #     self._build_gaussian_pyramid()
    #     # You can also change to your own base pyramid
    #     # self._build_user_designed_base_pyramid()
    #
    # @override
    # def _init_after_change_image(self) -> None:
    #     """
    #     Initializes after set a image.
    #     """
    #     self._build_base_pyramid()
    #     # You can do others after set an image

    def decomposition_ordinary(self) -> None:
        """
        Perform decomposition of the pyramid using ordinary techniques.

        This method is specific to Demo class and implements the decomposition process
        using ordinary techniques.
        """
        pass

    # def decomposition_your_method(self) -> None:
    #     """
    #     You can just change the name of funtion to adapt the `self.decom_way`
    #     """
    #     pass

    def reconstruction_ordinary(self) -> None:
        """
        Perform reconstruction of the pyramid using ordinary techniques.

        This method is specific to Demo class and implements the reconstruction process
        using ordinary techniques.
        """
        pass

    # def reconstruction_your_method(self) -> None:
    #     """
    #     You can just change the name of funtion to adapt the `self.recon_way`
    #     """
    #     pass


class Laplacian(Base):
    """
    Laplacian pyramid

    This class provides methods for decomposition and reconstruction of the Laplacian pyramid
    using ordinary and orthogonal techniques.

    Reference:
        P. Burt and E. Adelson, "The Laplacian Pyramid as a Compact Image Code," 
        in IEEE Transactions on Communications, vol. 31, no. 4, pp. 532-540, 
        April 1983, doi: 10.1109/TCOM.1983.1095851.
    """
    def __init__(self,**kwargs) -> None:
        super().__init__("Laplacian",**kwargs)

    def decomposition_ordinary(self) -> list[torch.Tensor]:
        """
        Perform decomposition of the Laplacian pyramid using ordinary techniques.

        This method computes the Laplacian pyramid by subtracting each layer of the Gaussian pyramid
        from the corresponding upsampled layer of the Gaussian pyramid.
        """
        laplacian_pyramid = []
        for i in range(self.layer):
            _,_,m,n = self.gaussian[i].shape
            expanded = self.pyr_up(self.gaussian[i+1])[:,:,:m,:n]
            laplacian = self.gaussian[i] - expanded
            laplacian_pyramid.append(laplacian)

        self.pyramid = laplacian_pyramid
        return self.pyramid

    def reconstruction_ordinary(self) -> torch.Tensor:
        """
        Perform reconstruction of the Laplacian pyramid using ordinary techniques.

        This method reconstructs the image from the Laplacian pyramid using ordinary reconstruction
        techniques, which involves adding each layer of the Laplacian pyramid to the corresponding
        upsampled version of the reconstructed image.
        """
        image_reconstructed = self.gaussian[-1]
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            expanded = self.pyr_up(image_reconstructed)[:,:,:m,:n]
            image_reconstructed = self.pyramid[i] + expanded

        self.recon = image_reconstructed
        return self.recon

    def reconstruction_orthogonal(self) -> torch.Tensor:
        """
        Perform reconstruction of the Laplacian pyramid using orthogonal techniques.

        This method reconstructs the image from the Laplacian pyramid using orthogonal reconstruction
        techniques, which involves subtracting each downsampled version of the Laplacian pyramid from
        the reconstructed image, and then adding each layer of the Laplacian pyramid to the corresponding
        upsampled version of the reconstructed image.

        Reference:
        M. N. Do and M. Vetterli, "Framing pyramids," in IEEE Transactions on Signal Processing, 
        vol. 51, no. 9, pp. 2329-2342, Sept. 2003, doi: 10.1109/TSP.2003.815389.
        """
        image_reconstructed = self.gaussian[-1]
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            image_reconstructed -= self.pyr_down(self.pyramid[i])
            expanded = self.pyr_up(image_reconstructed)[:,:,:m,:n]
            image_reconstructed = self.pyramid[i] + expanded

        self.recon = image_reconstructed
        return self.recon


class Contrust(Base):
    """
    Contrast pyramid

    This class provides methods for decomposition and reconstruction of the Contrast pyramid
    using ordinary techniques.

    Reference:
        Toet, Alexander et al. “Merging thermal and visual images by a contrast pyramid.” 
        Optical Engineering 28 (1989): 789-792.
    """
    def __init__(self,**kwargs) -> None:
        super().__init__("Contrust",**kwargs)

    def decomposition_ordinary(self) -> list[torch.Tensor]:
        """
        Perform decomposition of the Contrast pyramid using ordinary techniques.

        This method computes the Contrast pyramid by dividing each layer of the Gaussian pyramid
        by the corresponding upsampled layer of the Gaussian pyramid, subtracting 1, and replacing
        zeros in the denominator with zeros.
        """
        laplacian_pyramid = []
        for i in range(self.layer):
            _,_,m,n = self.gaussian[i].shape
            expanded = self.pyr_up(self.gaussian[i+1])[:,:,:m,:n]
            laplacian = torch.where(expanded == 0, torch.zeros_like(self.gaussian[i]),\
                self.gaussian[i] / expanded - 1)
            laplacian_pyramid.append(laplacian)

        self.pyramid = laplacian_pyramid
        return self.pyramid

    def reconstruction_ordinary(self) -> torch.Tensor:
        """
        Perform reconstruction of the Contrast pyramid using ordinary techniques.

        This method reconstructs the image from the Contrast pyramid using ordinary reconstruction
        techniques, which involves multiplying each layer of the Contrast pyramid by the corresponding
        upsampled version of the reconstructed image, and then adding 1.
        """
        image_reconstructed = self.gaussian[-1]
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            expanded = self.pyr_up(image_reconstructed)[:,:,:m,:n]
            image_reconstructed = (self.pyramid[i] + 1) * expanded

        self.recon = image_reconstructed
        return self.recon


class FSD(Base):
    """
    FSD (Filter Subtract Decimate) pyramid

    It can be regarded as a fast and improved version of the Laplacian pyramid.

    Reference:
        Hahn, M., & Samadzadegan, F. (2004, July). A study of image fusion techniques in remote sensing. 
        In Proc. 20th ISPRS Congress Geoimagery Bridging Continents (pp. 889-895).
    """
    def __init__(self,**kwargs) -> None:
        super().__init__("FSD",**kwargs)

    def decomposition_ordinary(self) -> List[torch.Tensor]:
        """
        Perform decomposition of the FSD pyramid using ordinary techniques.

        * Laplaian: Li = Gi - expand(subsample(gaussian_blur(Gi)))
        * FSD:      Li = Gi - gaussian_blur(Gi) <- simlified
        """
        fsd_pyramid = []
        for i in range(self.layer):
            fsd = self.gaussian[i] - self.gaussian_blur(self.gaussian[i])
            fsd_pyramid.append(fsd)

        self.pyramid = fsd_pyramid
        return fsd_pyramid

    def reconstruction_ordinary(self) -> torch.Tensor:
        """
        Same as Laplaian ordinary.
        """
        image_reconstructed = self.gaussian[-1]
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            expanded = self.pyr_up(image_reconstructed)[:,:,:m,:n]
            image_reconstructed = self.pyramid[i] + expanded

        self.recon = image_reconstructed
        return self.recon


class Graident(Base):
    """
    Gradient pyramid
    """
    def __init__(self,**kwargs) -> None:
        super().__init__("Graident",**kwargs)

    @staticmethod
    def get_graident(image: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Compute gradients of the input image.

        Args:
            image (torch.Tensor or list): Input image tensor or list of tensors.

        Returns:
            list: List of gradient tensors.
        """
        if not isinstance(image, list):
            image = [image]*4
        h1 = torch.tensor([[0,0,0],[0, 1,-1],[0,0,0]], dtype=torch.float32)
        h2 = torch.tensor([[0,0,0],[0, 0,-1],[0,1,0]], dtype=torch.float32) / torch.sqrt(torch.tensor(2))
        h3 = torch.tensor([[0,0,0],[0,-1, 0],[0,1,0]], dtype=torch.float32)
        h4 = torch.tensor([[0,0,0],[0,-1, 0],[0,0,1]], dtype=torch.float32) / torch.sqrt(torch.tensor(2))
        h = [k.unsqueeze(0).unsqueeze(0).repeat(image[0].shape[1], 1, 1, 1).to(image[0].dtype) for k in [h1,h2,h3,h4]]
        return [F.conv2d(_i, _h, stride=1, padding=1, groups=_i.shape[1]) for _i,_h in zip(image,h)]

    def decomposition_ordinary(self) -> None:
        """
        Perform decomposition of the Gradient pyramid using ordinary techniques.

        This method computes the Gradient pyramid by applying Gaussian blur to each layer
        of the Gaussian pyramid, adding it back to the original layer, and then computing
        gradients.

        Returns:
            None
        """
        graident_pyramid = []
        for i in range(self.layer):
            temp = self.gaussian_blur(self.gaussian[i], kernel_size=3) + self.gaussian[i]
            graident_pyramid.append(self.get_graident(temp))
        self.pyramid = graident_pyramid

    def reconstruction_ordinary(self) -> None:
        """
        Perform reconstruction of the Gradient pyramid using ordinary techniques.

        This method reconstructs the image from the Gradient pyramid using ordinary
        reconstruction techniques, which involves computing the negative sum of gradients
        and adding it to an upsampled version of the reconstructed image.

        Returns:
            None
        """
        lp = Laplacian(image=self.image,auto=False,layer=self.layer)
        for i in range(self.layer):
            temp = torch.stack(self.get_graident(self.pyramid[i]))
            temp = -torch.sum(temp, dim=0) / 8
            pyramid = self.gaussian_blur(temp,kernel_size=3,bias=1) # change FSD to Laplacian
            lp.append(pyramid)
        lp.reconstruction()
        self.recon = lp.recon


class Morphological(Base):
    """
    Represents a pyramid constructed using morphological operations on an input image.

    Attributes:
        name (str): Name of the pyramid.
        image (torch.Tensor): Input image tensor.
        pyramid (List[torch.Tensor]): List to store pyramid layers.
        layer (int): Number of layers in the pyramid.
        recon (torch.Tensor): Output image after reconstruction.
        auto (bool): Flag indicating whether to automatically construct and reconstruct the pyramid.
        down_way (str): Method used for downsampling during pyramid construction.
    """
    def __init__(self,**kwargs) -> None:
        """
        Initializes a Morphological pyramid object.

        Args:
            **kwargs: Additional keyword arguments to customize object attributes.
        """
        super().__init__("Morphological",**kwargs)

    @staticmethod
    def morph_dilation(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Applies dilation operation to the input image.

        Args:
            image (torch.Tensor): Input image tensor.
            kernel_size (int): Size of the dilation kernel.

        Returns:
            torch.Tensor: Dilated image tensor.
        """
        # Define the kernel for dilation
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

        # Perform dilation operation
        dilated_image = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        return dilated_image

    @staticmethod
    def morph_erosion(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Applies erosion operation to the input image.

        Args:
            image (torch.Tensor): Input image tensor.
            kernel_size (int): Size of the erosion kernel.

        Returns:
            torch.Tensor: Eroded image tensor.
        """
        # Define the kernel for erosion
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

        # Perform erosion operation
        eroded_image = F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        return eroded_image

    @staticmethod
    def morph_opening(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Performs opening operation on the input image.

        Args:
            image (torch.Tensor): Input image tensor.
            kernel_size (int): Size of the kernel.

        Returns:
            torch.Tensor: Opened image tensor.
        """
        # Perform erosion followed by dilation (opening)
        eroded_image = Morphological.morph_erosion(image, kernel_size=kernel_size)
        opened_image = Morphological.morph_dilation(eroded_image, kernel_size=kernel_size)

        return opened_image

    @staticmethod
    def morph_closing(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Performs closing operation on the input image.

        Args:
            image (torch.Tensor): Input image tensor.
            kernel_size (int): Size of the kernel.

        Returns:
            torch.Tensor: Closed image tensor.
        """
        # Perform dilation followed by erosion (closing)
        dilated_image = Morphological.morph_dilation(image, kernel_size=kernel_size)
        closed_image = Morphological.morph_erosion(dilated_image, kernel_size=kernel_size)

        return closed_image

    def _build_morphology_pyramid(self) -> None:
        """
        Constructs a Morphology pyramid from the input image.
        """
        if self.image is not None:
            image = self.image
            self.morphology = [image]
            for _ in range(self.layer):
                image  = self.morph_opening(image)
                image = self.morph_closing(image)
                image = self.down_sample(image)
                self.morphology.append(image)
        else:
            raise ValueError("You should first assign a image.")

    @override
    def _build_base_pyramid(self) -> None:
        self._build_morphology_pyramid()

    def decomposition_ordinary(self) -> None:
        """
        Performs ordinary decomposition of the pyramid.
        """
        morph_pyramid = []
        for i in range(self.layer):
            _,_,m,n = self.morphology[i].shape
            expanded = self.up_sample(self.morphology[i+1])[:,:,:m,:n]
            expanded = self.morph_closing(expanded)
            expanded = self.morph_opening(expanded)
            morph = self.morphology[i] - expanded
            morph_pyramid.append(morph)

        self.pyramid = morph_pyramid

    def reconstruction_ordinary(self) -> None:
        """
        Performs ordinary reconstruction of the pyramid.
        """
        image_reconstructed = self.morphology[-1]
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            expanded = self.up_sample(image_reconstructed)[:,:,:m,:n]
            expanded = self.morph_closing(expanded)
            expanded = self.morph_opening(expanded)
            image_reconstructed = self.pyramid[i] + expanded

        self.recon = image_reconstructed


def test_laplacian():
    # Gray & ordinary construction
    pyramid = Laplacian(image=ir)
    glance(pyramid.pyramid)
    glance([ir,pyramid.recon])
    # Comment: Good

    # Gray & orthogonal construction
    pyramid = Laplacian(image=vis,recon_way='orthogonal')
    glance(pyramid.pyramid)
    glance([vis,pyramid.recon])
    # Comment: Especially for blued images

def test_contrust():
    pyramid = Contrust(image=ir)
    glance(pyramid.pyramid)
    glance([ir,pyramid.recon])
    # Comment: better than laplacian for fusion

def test_fsd():
    # layer=5
    pyramid = FSD(image=ir)
    glance(pyramid.pyramid)
    glance([ir,pyramid.recon]) # One Bad point!

    # layer=3
    pyramid = FSD(image=ir,layer=3)
    glance(pyramid.pyramid)
    glance([ir,pyramid.recon]) # Many bad points!!!

    # layer=6
    pyramid = FSD(image=ir,layer=6)
    glance(pyramid.pyramid)
    glance([ir,pyramid.recon]) # No bad points~
    # Comment: bad, sacrifice quality for speed

def test_graident():
    pyramid = Graident(image=ir)
    # glance(pyramid.pyramid)
    glance([ir,pyramid.recon])
    # Comment: bad

def test_morphological():
    pyramid = Morphological(image=ir)
    glance(pyramid.pyramid)
    glance([ir,pyramid.recon])
    # Comment: good

if __name__ == '__main__':
    from clib.metrics.fusion import ir,vis
    from clib.utils import glance
    # test_laplacian()
    # test_contrust()
    # test_fsd()
    # test_graident()
    # test_morphological()