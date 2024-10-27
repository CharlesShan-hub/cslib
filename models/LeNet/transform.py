from torchvision.transforms import Compose, Resize, ToTensor, Normalize

transform = Compose([Resize((28, 28)), ToTensor(), Normalize((0.5,), (0.5,))])
