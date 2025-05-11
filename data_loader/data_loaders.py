from torchvision import datasets, transforms
from base import BaseDataLoader

class SelfSupervisedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, image  # input and target are the same
    

class Gum_Dataloader(BaseDataLoader):


    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((134, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.2788, 0.2573, 0.2250), (0.3075, 0.2872, 0.2669)) #Sampled from data
        ])
        self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = SelfSupervisedImageFolder(root=self.data_dir, transform=trsfm)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
