from typing import List, Union, Optional
from typing_extensions import override
import torch
# import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np

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
        gau_blur_way: str = 'Paper', sigma: Optional[List[float]] = None, bias: float = 0) -> torch.Tensor:
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
                kernel = torch.tensor([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]], dtype=torch.float32) / 16 + bias
            elif kernel_size == 5:
                kernel = torch.tensor([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]], dtype=torch.float32) / 256 + bias
            else:
                raise ValueError(f"kernel size in paper only be 3 or 5, not {kernel_size}")
            # Expand dimensions of the kernel for convolution
            kernel = kernel.unsqueeze(0).unsqueeze(0)

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
        if self.image is not None:
            image = self.image
            _, _, width, height = image.shape
            if self.layer > int(torch.floor(torch.log2(torch.tensor(min(width, height)))) - 2):
                raise RuntimeError('Cannot build {} levels, image too small.'.format(self.layer))
            self.gaussian = [image]
            for _ in range(self.layer):
                image = self.pyr_down(image)
                self.gaussian.append(image)
        else:
            raise ValueError("You should first assign a image.")

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
    Represents a Laplacian pyramid, derived from the Base class.

    This class provides methods for decomposition and reconstruction of the Laplacian pyramid
    using ordinary and orthogonal techniques.

    Attributes:
        Inherits attributes from the Base class.
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
    Represents a Contrast pyramid, derived from the Base class.

    This class provides methods for decomposition and reconstruction of the Contrast pyramid
    using ordinary techniques.

    Attributes:
        Inherits attributes from the Base class.
    """
    def __init__(self,**kwargs) -> None:
        """
        Initializes a Contrast object.

        Args:
            **kwargs: Additional keyword arguments to customize object attributes.

        Inherits attributes from the Base class.
        """
        super().__init__("Contrust",**kwargs)

    def decomposition_ordinary(self) -> None:
        """
        Perform decomposition of the Contrast pyramid using ordinary techniques.

        This method computes the Contrast pyramid by dividing each layer of the Gaussian pyramid
        by the corresponding upsampled layer of the Gaussian pyramid, subtracting 1, and replacing
        zeros in the denominator with zeros.

        Returns:
            None
        """
        laplacian_pyramid = []
        for i in range(self.layer):
            _,_,m,n = self.gaussian[i].shape
            expanded = self.pyr_up(self.gaussian[i+1])[:,:,:m,:n]
            laplacian = torch.where(expanded == 0, torch.zeros_like(self.gaussian[i]),\
                self.gaussian[i] / expanded - 1)
            laplacian_pyramid.append(laplacian)

        self.pyramid = laplacian_pyramid

    def reconstruction_ordinary(self) -> None:
        """
        Perform reconstruction of the Contrast pyramid using ordinary techniques.

        This method reconstructs the image from the Contrast pyramid using ordinary reconstruction
        techniques, which involves multiplying each layer of the Contrast pyramid by the corresponding
        upsampled version of the reconstructed image, and then adding 1.

        Returns:
            None
        """
        image_reconstructed = self.gaussian[-1]
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            expanded = self.pyr_up(image_reconstructed)[:,:,:m,:n]
            image_reconstructed = (self.pyramid[i] + 1) * expanded

        self.recon = image_reconstructed


class FSD(Base):
    """
    Represents a FSD (Feature Suppression Decomposition) pyramid, derived from the Base class.

    This class provides methods for decomposition and reconstruction of the FSD pyramid
    using ordinary techniques.

    Attributes:
        Inherits attributes from the Base class.
    """
    def __init__(self,**kwargs) -> None:
        """
        Initializes a FSD object.

        Args:
            **kwargs: Additional keyword arguments to customize object attributes.

        Inherits attributes from the Base class.
        """
        super().__init__("FSD",**kwargs)

    def decomposition_ordinary(self) -> None:
        """
        Perform decomposition of the FSD pyramid using ordinary techniques.

        This method computes the FSD pyramid by subtracting the Gaussian-blurred version of each layer
        from the corresponding layer of the Gaussian pyramid.

        Returns:
            None
        """
        fsd_pyramid = []
        for i in range(self.layer):
            _,_,m,n = self.gaussian[i].shape
            fsd = self.gaussian[i] - self.gaussian_blur(self.gaussian[i])
            fsd_pyramid.append(fsd)

        self.pyramid = fsd_pyramid

    def reconstruction_ordinary(self) -> None:
        """
        Perform reconstruction of the FSD pyramid using ordinary techniques.

        This method reconstructs the image from the FSD pyramid using ordinary reconstruction
        techniques, which involves iteratively adding the FSD layers to an upsampled version of
        the reconstructed image, blurring the result, and then repeating the process.

        Returns:
            None
        """
        image_reconstructed = self.up_sample(self.gaussian[-1])*4
        for i in reversed(range(self.layer)):
            _,_,m,n = self.pyramid[i].shape
            image_reconstructed = image_reconstructed[:,:,:m,:n]
            image_reconstructed += self.pyramid[i]
            image_reconstructed = self.gaussian_blur(image_reconstructed)
            image_reconstructed += self.pyramid[i]
            if i!=0:
                image_reconstructed = self.up_sample(image_reconstructed)*4

        self.recon = image_reconstructed


class Graident(Base):
    """
    Represents a Gradient pyramid, derived from the Base class.

    This class provides methods for decomposition and reconstruction of the Gradient pyramid
    using ordinary techniques.

    Attributes:
        Inherits attributes from the Base class.
    """
    def __init__(self,**kwargs) -> None:
        """
        Initializes a Gradient object.

        Args:
            **kwargs: Additional keyword arguments to customize object attributes.

        Inherits attributes from the Base class.
        """
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
        h = [k.unsqueeze(0).unsqueeze(0).repeat(image[0].shape[1], 1, 1, 1) for k in [h1,h2,h3,h4]]
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


# class Steerable(Base):
#     """
#     Ref: https://github.com/tomrunia/PyTorchSteerablePyramid
#     """
#     def __init__(self, **kwargs) -> None:
#         self.nbands = 4 # number of orientation bands
#         self.scale_factor = 2
#         super().__init__("Steerable", **kwargs)

#     @staticmethod
#     def prepare_grid(m, n):
#         x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m, num=m)
#         y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n, num=n)
#         xv, yv = np.meshgrid(y, x)
#         angle = np.arctan2(yv, xv)
#         rad = np.sqrt(xv**2 + yv**2)
#         rad[m//2][n//2] = rad[m//2][n//2 - 1]
#         log_rad = np.log2(rad)
#         return log_rad, angle

#     @staticmethod
#     def rcosFn(width, position):
#         N = 256  # abritrary
#         X = np.pi * np.array(range(-N-1, 2))/2/N
#         Y = np.cos(X)**2
#         Y[0] = Y[1]
#         Y[N+2] = Y[N+1]
#         X = position + 2*width/np.pi*(X + np.pi/4)
#         return X, Y

#     @staticmethod
#     def pointOp(im, Y, X):
#         out = np.interp(im.flatten(), X, Y)
#         return np.reshape(out, im.shape)

#     @override
#     def _init_after_change_image(self) -> None:
#         """
#         Initializes after set a image.
#         """
#         # self._build_base_pyramid()
#         # Check Image
#         if self.image is None:
#             raise ValueError("No Image to do decomposition")
#         _, _, height, width = self.image.shape

#         # Prepare a grid
#         self.log_rad, self.angle = self.prepare_grid(height, width)

#         # Radial transition function (a raised cosine in log-frequency):
#         self.Xrcos, self.Yrcos = self.rcosFn(1, -0.5)
#         self.Yrcos = torch.sqrt(self.Yrcos)
#         self.YIrcos = torch.sqrt(1 - self.Yrcos**2)
#         self.lo0mask = self.pointOp(self.log_rad, self.YIrcos, self.Xrcos)
#         self.hi0mask = self.pointOp(self.log_rad, self.Yrcos, self.Xrcos)
#         # Note that we expand dims to support broadcasting later
#         self.lo0mask = torch.from_numpy(self.lo0mask).float()[None,:,:,None]
#         self.hi0mask = torch.from_numpy(self.hi0mask).float()[None,:,:,None]

#     def decomposition_ordinary(self) -> None:
#         """
#         Perform decomposition of the Laplacian pyramid using ordinary techniques.

#         This method computes the Laplacian pyramid by subtracting each layer of the Gaussian pyramid
#         from the corresponding upsampled layer of the Gaussian pyramid.

#         Returns:
#             None
#         """

#         steerable_pyramid = []
#         for i in range(self.layer):
#         #     _,_,m,n = self.gaussian[i].shape
#         #     expanded = self.pyr_up(self.gaussian[i+1])[:,:,:m,:n]
#         #     laplacian = self.gaussian[i] - expanded
#         #     laplacian_pyramid.append(laplacian)

#         # self.pyramid = laplacian_pyramid

#     def reconstruction_ordinary(self) -> None:
#         """
#         Perform reconstruction of the Laplacian pyramid using ordinary techniques.

#         This method reconstructs the image from the Laplacian pyramid using ordinary reconstruction
#         techniques, which involves adding each layer of the Laplacian pyramid to the corresponding
#         upsampled version of the reconstructed image.

#         Returns:
#             None
#         """
#         image_reconstructed = self.gaussian[-1]
#         for i in reversed(range(self.layer)):
#             _,_,m,n = self.pyramid[i].shape
#             expanded = self.pyr_up(image_reconstructed)[:,:,:m,:n]
#             image_reconstructed = self.pyramid[i] + expanded

#         self.recon = image_reconstructed

