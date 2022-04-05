import os
import random
import numpy as np
import pickle
import torch
from torch.utils.data.dataset import Dataset


class AsanDataset(Dataset):
    def __init__(self, data_path, transform=None):
        filenames = sorted(os.listdir(data_path))
        self.file_paths = [os.path.join(data_path, f) for f in filenames]
        self.transform = transform

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'rb') as f_data:
            data = pickle.load(f_data)
        # data is a dictionary containing the following keys:
        #   image : Image data [?, ?] (1, G, 512, 512) numpy int16
        #   label : GT label  (G, 512, 512) numpy uint8
        img = torch.from_numpy(data['image'])
        lbl = torch.from_numpy(data['label'])
        
        if self.transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def __len__(self):
        return len(self.file_paths)