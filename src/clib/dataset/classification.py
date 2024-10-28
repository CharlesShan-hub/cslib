from .collection.flowers17 import Flowers17
from torchvision.datasets import \
    Caltech101, Caltech256, \
    CelebA, \
    CIFAR10, CIFAR100, \
    Country211, \
    DTD, \
    MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST, \
    EuroSAT, \
    FakeData, \
    FER2013, \
    FGVCAircraft, \
    Flickr8k, Flickr30k, \
    Flowers102, \
    Food101, \
    GTSRB, \
    INaturalist, \
    ImageNet, \
    LFWPeople, \
    LSUN, \
    Omniglot, \
    OxfordIIITPet, \
    Places365, \
    PCAM, \
    RenderedSST2, \
    SEMEION, \
    SBU, \
    StanfordCars, \
    STL10, \
    SUN397, \
    SVHN
# USP
# Imagenette, \

# from torch.utils.data import Dataset
# from torchvision import transforms
# from pathlib import Path

# class Flowers17(Dataset):
#     def __init__(self,
#                  root_dir: Union[str, Path], 
#                  transform = transforms.Compose([transforms.ToTensor()])
#         ) -> None:
#         super().__init__()
#         self.root_dir = Path(root_dir)
#         self.transform = transform
#         with open(self.root_dir / 'jpg/files.txt') as f:
#             img_path = f.readlines()

#         # Check
#         if check:
#             for i,j in zip(self.ir_paths, self.vis_paths):
#                 assert i.name == j.name
        
#     def __len__(self):
#         return len(self.ir_paths)

#     def __getitem__(self, idx):
#         # Load images
#         ir_image = self.ir_paths[idx].__str__()
#         vis_image = self.vis_paths[idx].__str__()
#         img_id, _ = os.path.splitext(self.ir_paths[idx].name)

#         # Apply transform if specified
#         if self.only_path == False:
#             ir_image = self.transform(Image.open(ir_image))
#             vis_image = self.transform(Image.open(vis_image))

#         # Return a dictionary with all images
#         sample = {
#             'ir': ir_image,
#             'vis': vis_image,
#             'id': img_id
#         }
#         return sample


