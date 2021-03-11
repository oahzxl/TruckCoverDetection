import os
import warnings
from glob import glob

import PIL
import cv2
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from utils.transformer import *

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size, class2idx, mode=None, transforms=None):
        self.mode = mode
        self.class2idx = class2idx
        if self.mode == 'test':
            self.img_list = glob(os.path.join(img_path, "*.jpg"))
            self.img_list += glob(os.path.join(img_path, "*.png"))
        else:
            self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
            self.img_list += glob(os.path.join(img_path, "*", "*.png"))
        if transforms is None:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=input_size),  # 缩放
                # Resize(size=input_size),  # 等比填充缩放
                # torchvision.transforms.RandomCrop(size=input_size),
                torchvision.transforms.RandomResizedCrop(size=input_size, scale=(0.7, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                RandomGaussianBlur(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img = PIL.Image.open(self.img_list[idx])
            if img.format == 'PNG' or img.format == 'GIF':
                img = img.convert("RGB")
            elif img.layers == 1:
                img = cv2.imread(self.img_list[idx], 0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transforms(img)
        except OSError:
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        if self.mode == 'train' or self.mode == 'eval':
            if self.img_list[idx].split('/')[-2] in self.class2idx:
                label = self.class2idx[self.img_list[idx].split('/')[-2]]
            else:
                print(self.img_list[idx].split('/')[-2])
                raise ValueError
            sample = {"image": img, "label": label, "name": self.img_list[idx].split('/')[-1]}
        else:
            sample = {"image": img, "name": self.img_list[idx].split('/')[-1]}

        return sample


def build_dataset(data_path, input_size, mode, class2idx):
    return BinaryClassifierDataset(os.path.join(data_path, mode), input_size, mode=mode, class2idx=class2idx)
