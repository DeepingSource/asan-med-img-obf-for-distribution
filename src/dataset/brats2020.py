"""Credit: https://github.com/akug/oriu-brats15/blob/master/code/load.py """
import os
import random
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset


class Brats2020(Dataset):
    """ Dataset class to load the 3D brain images of the BraTS2015 challenge (original data)
    """
    def __init__(self, path, train, train_split=0.8, grouping=128, transform=None):
        """
        Args:
            path (str): Should point to a folder named BraTS2015_Training
            train (bool): If True, load training data, if False load testing data
            train_split (float): Ratio of training to validation split
            transform: Transformation done on the input images
        """

        train_path = os.path.join(path, 'preprocessed/group_{}/'.format(grouping), 'train')
        assert os.path.isdir(train_path), train_path

        self.transform = transform
        self.paths = []

        random.seed(111)
        paths = sorted(os.listdir(train_path))
        random.shuffle(self.paths)
        # Split by scans
        if train:
            paths = paths[:int(train_split*len(paths))]
        else:
            paths = paths[int(train_split*len(paths)):]
        
        for i in range(len(paths)):
            self.paths.append(os.path.join(train_path, paths[i]))
        

    def __getitem__(self, idx):
        imgs_npy = np.load(self.paths[idx])
        img = torch.from_numpy(imgs_npy[:-1])
        lbl = torch.from_numpy(imgs_npy[-1]).to(torch.long)

        if self.transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def __len__(self):
        return len(self.paths)