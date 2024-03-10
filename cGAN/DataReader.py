from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            pics = line.split()
            imgs.append((pics[0], pics[1]))
            self.transform = transform
            self.target_transform = target_transform
        self.imgs = imgs

    def __getitem__(self, index):
        p1, p2 = self.imgs[index]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            img2 = self.target_transform(img2)
        return (img1, img2)

    def __len__(self):
        return len(self.imgs)