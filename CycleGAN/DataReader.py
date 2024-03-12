from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import os

class MyDataset(Dataset):
    def __init__(self, path, num, transform = None):
        self.num = num
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        path = os.path.join(self.path, '%05d.jpg' % (index + 1))
        img = Image.open(path).convert('RGB')
        if(self.transform is not None):
            img = self.transform(img)
        return img

    def __len__(self):
        return self.num