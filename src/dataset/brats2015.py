"""Credit: https://github.com/akug/oriu-brats15/blob/master/code/load.py """
import os
import random
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

SCAN_TYPES = ['MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair', 'all']
GLIOMA_TYPES = ['HGG', 'LGG', 'all']

class Brats2015(Dataset):
    """ Dataset class to load the 3D brain images of the BraTS2015 challenge (original data)
    """
    def __init__(self, path, train, train_split=0.8, glioma_type='all', scan_type='all', grouping=128, transform=None):
        """
        Args:
            path (str): Should point to a folder named BraTS2015_Training
            train (bool): If True, load training data, if False load testing data
            train_split (float): Ratio of training to validation split
            gg (str): Can be 'HGG', 'LGG' or 'all'
            scan_type (str): Can be 'MR_T1', 'MR_T1c', 'MR_T2', 'MR_Flair' or 'all'
            transform: Transformation done on the input images
        """

        assert glioma_type in GLIOMA_TYPES
        assert scan_type in SCAN_TYPES
        train_path = os.path.join(path, 'preprocessed/group_{}/'.format(grouping), 'train')
        assert os.path.isdir(train_path)

        self.transform = transform
        self.paths = []
        self.scan_type = scan_type

        scan_paths = []
        
        if glioma_type == 'all':
            glioma_type = ['HGG', 'LGG']
        else:
            glioma_type = [glioma_type]
        
        for g_type in glioma_type:
            type_path = os.path.join(train_path, g_type) # data_preprocessed/group_8/train/HGG
            for pp in os.listdir(type_path):
                scan_path = os.path.join(type_path, pp) # data_preprocessed/group_8/train/HGG/brats_tcia_pat156_0001
                scan_paths.append(scan_path)

        random.seed(111)
        scan_paths = sorted(scan_paths)
        random.shuffle(scan_paths)
        # Split by scans
        if train:
            scan_paths = scan_paths[:int(train_split*len(scan_paths))]
        else:
            scan_paths = scan_paths[int(train_split*len(scan_paths)):]
        
        # Collect scan slices
        for sp in scan_paths:
            for scan_slice in os.listdir(sp):
                scan_slice_path = os.path.join(sp, scan_slice)
                self.paths.append(scan_slice_path)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f_data:
            scan_dict = pickle.load(f_data)
        if self.scan_type == 'all':
            img = torch.from_numpy(np.concatenate((scan_dict['MR_Flair'], 
                                                scan_dict['MR_T1'],
                                                scan_dict['MR_T1c'],
                                                scan_dict['MR_T2']), axis=0))
            lbl = torch.from_numpy(scan_dict['OT'])
        else:
            raise NotImplementedError
        
        if self.transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def __len__(self):
        return len(self.paths)


class Brats2015Test(Dataset):
    """ Dataset class to load the 3D brain images of the BraTS2015 challenge (original data)
    """

    def __init__(self, path, grouping=16, ):
        """
        Args:
            path (str): Should point to a folder named BraTS2015_Training
        """
        train_path = os.path.join(path, 'preprocessed/group_{}/'.format(grouping), 'test')
        assert os.path.isdir(train_path)

        self.paths = []

        scan_paths = []

        glioma_type = ['HGG_LGG']

        for g_type in glioma_type:
            type_path = os.path.join(train_path, g_type)  # data_preprocessed/group_8/train/HGG
            for pp in os.listdir(type_path):
                scan_path = os.path.join(type_path, pp)  # data_preprocessed/group_8/train/HGG/brats_tcia_pat156_0001
                scan_paths.append(scan_path)

        scan_paths = sorted(scan_paths)

        # Collect scan slices
        for sp in scan_paths:
            for scan_slice in os.listdir(sp):
                scan_slice_path = os.path.join(sp, scan_slice)
                self.paths.append(scan_slice_path)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f_data:
            scan_dict = pickle.load(f_data)
        img = torch.from_numpy(np.concatenate((scan_dict['MR_Flair'],
                                               scan_dict['MR_T1'],
                                               scan_dict['MR_T1c'],
                                               scan_dict['MR_T2']), axis=0))

        return img, self.paths[idx]

    def __len__(self):
        return len(self.paths)