from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Gum_Dataloader(BaseDataLoader):


    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((134, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.2788, 0.2573, 0.2250), (0.3075, 0.2872, 0.2669)) #No idea what these arebitrary values are #(tensor([0.2788, 0.2573, 0.2250]), tensor([0.3075, 0.2872, 0.2669]))
        ])
        self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=trsfm)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
