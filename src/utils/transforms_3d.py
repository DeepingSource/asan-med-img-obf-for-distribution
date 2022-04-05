import torch
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class DownScale(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, target):
        """
            image:  (N x C x D x H x W)
            target: (N x D x H x W)
        """
        image_shape = image.shape
        target_shape = target.shape
        target_size = (image_shape[-2]//self.factor, image_shape[-1]//self.factor)
        image_resize = torch.zeros(size=(*image_shape[0:2], *target_size), dtype=image.dtype)
        target_resize = torch.zeros((target_shape[0], *target_size), dtype=target.dtype)
        for idx in range(image_shape[1]):
            image_resize[:,idx] = image[:, idx, ::self.factor, ::self.factor]
            target_resize[idx] = target[idx, ::2, ::2]
        return image_resize, target_resize

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        if np.random.rand() >= 0.5:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        return image, target

class RandomCrop(object):
    def __init__(self, size, padding):
        self.size = size
        self.padding = padding

    def __call__(self, image, target):
        image_shape = image.shape
        target_shape = target.shape
        # Apply Padding
        image_big = F.pad(image, pad=[self.padding] * 4)
        target_big = F.pad(target, pad=[self.padding] * 4)
        # Select random range
        rndx, rndy = torch.randint(low=0, high=self.padding * 2 , size=(2,))
        image = image_big[:, :, rndx:rndx+image_shape[-2], rndy:rndy+image_shape[-1]]
        target = target_big[:, rndx:rndx+target_shape[-2], rndy:rndy+target_shape[-1]]
        return image, target