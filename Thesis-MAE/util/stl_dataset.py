from torchvision.datasets import STL10
from torch.utils.data import Dataset, random_split
import numpy as np
from PIL import Image

class STL10_Dataset(Dataset):
    def __init__(self, gray = False, transform=None):
        self.transform = transform
        self.unlabeled = STL10(root = './data', split = 'unlabeled', download = True)
        self._train = True
        self.limit = 5000 # limit the number of images
    
    def __len__(self):
        return min(len(self.unlabeled), self.limit)
    
    def __getitem__(self, idx):
        img, cls = self.unlabeled[idx]
        # print(type(img))
        # x = np.array(img)
        x = self.transform(img)
        return x
    
    def __str__(self):
        return f'STL10_Dataset with {self.__len__()} images'
    
    def __repr__(self):
        return f'STL10_Dataset, transform={self.transform}'