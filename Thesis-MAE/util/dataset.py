import os
import cv2
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset


class UnlabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.tif')]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.image_files[index])
        image = tiff.imread(image_name)
        # print(image.shape)
        image = normalize(image)
        # print(image.max(), image.min())
        image = self.transform(image)     # torch.size([1, 224, 224]), with values in around [-2.2, 2.2]
        # print("after transform: ", image.max(), image.min())
        return image
    
    def __str__(self):
        return f'UnlabelDataset with {self.__len__()} images'

    def __repr__(self):
        return f'UnlabelDataset(data_path={self.root_dir}, transform={self.transform})'
    
    
def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    original_min = image.min()
    original_max = image.max()
    original_mean = image.mean()
    image = (image - original_min) / (original_max - original_min)  # in [0, 1]
    return image
    