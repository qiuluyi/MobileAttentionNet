import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

data_path = "./dataset/"
img_path = os.path.join(data_path, 'images')
lab_path = os.path.join(data_path, 'labels')


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, shape=(1920, 1080), transform=None, target_transform=None):

        self.images = []
        self.labels = []
        for each in os.listdir(images_path):
            img_name, _ = each.split('.')
            self.images.append(os.path.join(images_path, each))
            self.labels.append(os.path.join(labels_path, img_name + '_json/label.png'))
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape
        self.n_sampels = len(self.images)

    def __len__(self):
        return self.n_sampels

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img = cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, self.shape)
        lab = cv2.cvtColor(cv2.imread(self.labels[index]), cv2.COLOR_BGR2RGB)
        lab = cv2.resize(lab, self.shape)
        lab[lab > 0] = 1
        lab = lab.astype('uint8')
        lab = np.eye(2)[lab[:, :, 0]]  # one-hot编码
        lab = lab.astype('float32')  #
        lab = lab.transpose(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.transform(lab)
        a = 1
        return img, lab


class TestDataset(Dataset):
    def __init__(self, image_path, shape=(1920, 1080), transform=None):
        self.images = []
        self.shape = shape
        self.transform = transform
        for each in os.listdir(image_path):
            self.images.append(os.path.join(image_path, each))
        self.n_samples = len(self.images)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img = cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.shape)
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print("Over !")
