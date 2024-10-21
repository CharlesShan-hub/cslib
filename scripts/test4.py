import numpy as np
from PIL import Image
import torch
from skimage import color


def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    if len(image.shape) == 4:
        if image.shape[0] != 1:
            raise ValueError("Batch number should be 1.")
        image = image[0,:,:,:]
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0)
        if image.shape[-1] == 1:
            image = image[:,:,0]
    elif len(image.shape) != 2:
        raise ValueError("Image should be an image.")
    return image.detach().cpu().numpy()
    

def _tensor_to_image(image: torch.Tensor) -> Image.Image:
    image_array = _tensor_to_numpy(image)*255.0
    return _numpy_to_image(image_array)

def _image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)/255.0


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.tensor(image)


def _numpy_to_image(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        return Image.fromarray(image.astype(np.uint8), mode="RGB") 


def _numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.tensor(image)


a = np.array([[[100,200,255],[0,10,20],[110,200,255],[20,10,20]]])/255.0
print(a)
b = color.rgb2ycbcr(a)
# print(b)
c = color.ycbcr2rgb(b)
print(c)