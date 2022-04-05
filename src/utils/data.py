import os
import torch

from dataset.asan import AsanDataset
from dataset.brats2015 import Brats2015, Brats2015Test
from dataset.brats2020 import Brats2020

from utils.transforms_3d import DownScale, RandomHorizontalFlip, RandomCrop, Compose

from config.config import ASAN_DATA_PATH, BRATS2015_DATA_PATH, BRATS2020_DATA_PATH


def get_data_specs(dataset_name):
    if dataset_name == 'asan':
        num_classes = 4
        in_channels = 1
        image_size = (256, 256)

        # Data augmentation
        train_transform = Compose([
            DownScale(factor=2),
            RandomCrop(size=(256, 256), padding=32),
            RandomHorizontalFlip(),
        ])
        test_transform = Compose([
            DownScale(factor=2),
        ])
    elif dataset_name == 'brats2015':
        num_classes = 5
        in_channels = 4
        image_size = (120, 120)

        # Data augmentation
        train_transform = Compose([
            DownScale(factor=2),
            RandomCrop(size=(120, 120), padding=32),
            RandomHorizontalFlip(),
        ])
        test_transform = Compose([
            DownScale(factor=2),
        ])
    elif dataset_name == 'brats2020':
        num_classes = 5
        in_channels = 4
        image_size = (120, 120)

        # Data augmentation
        train_transform = Compose([
            DownScale(factor=2),
            RandomCrop(size=(120, 120), padding=32),
            RandomHorizontalFlip(),
        ])
        test_transform = Compose([
            DownScale(factor=2),
        ])
    else:
        raise ValueError
    return num_classes, in_channels, image_size, train_transform, test_transform


def get_dataset(dataset_name, train_transform=None, test_transform=None, grouping=128):
    if dataset_name == 'asan':
        path_data_train = os.path.join(ASAN_DATA_PATH, 'data_preprocessed/CHD_3D_Group{}'.format(grouping), 'train')
        data_train = AsanDataset(data_path=path_data_train, transform=train_transform)
        # path_data_valid = os.path.join(ASAN_DATA_PATH, 'data_preprocessed/CHD_3D_Group8', 'val')
        # data_valid = AsanDataset(data_path=path_data_valid)
        path_data_test = os.path.join(ASAN_DATA_PATH, 'data_preprocessed/CHD_3D_Group{}'.format(grouping), 'val')
        data_test = AsanDataset(data_path=path_data_test, transform=test_transform)
    elif dataset_name == 'brats2015':
        data_train = Brats2015(path=BRATS2015_DATA_PATH, train=True, transform=train_transform, grouping=grouping)
        data_test = Brats2015(path=BRATS2015_DATA_PATH, train=False, transform=test_transform, grouping=grouping)
    elif dataset_name == 'brats2020':
        data_train = Brats2020(path=BRATS2020_DATA_PATH, train=True, transform=train_transform, grouping=grouping)
        data_test = Brats2020(path=BRATS2020_DATA_PATH, train=False, transform=test_transform, grouping=grouping)
    else:
        raise ValueError
    return data_train, data_test


def get_testset(dataset_name, test_transform=None, grouping=128):
    if dataset_name == 'asan':
        path_data_test = os.path.join(ASAN_DATA_PATH, 'data_preprocessed/CHD_3D_Group8', 'test')
        data_test = AsanDataset(data_path=path_data_test, transform=test_transform)
    elif dataset_name == 'brats2015':
        data_test = Brats2015Test(path=BRATS2015_DATA_PATH, grouping=grouping)
    elif dataset_name == 'brats2020':
        # ToDo
        data_test = Brats2020(path=BRATS2020_DATA_PATH, train=False, transform=test_transform, grouping=grouping)
    else:
        raise ValueError
    return data_test