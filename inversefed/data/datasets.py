"""This is dataset.py from pytorch-examples.

Refer to

https://github.com/pytorch/examples/blob/master/super_resolution/dataset.py.
"""
import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from os.path import basename
from PIL import Image
from torchvision.datasets.folder import ImageFolder

import json


def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def _load_img(filepath, RGB=True):
    img = Image.open(filepath)
    if RGB:
        pass
    else:
        img = img.convert('YCbCr')
        img, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    """Generate an image-to-image dataset from images from the given folder."""

    def __init__(self, image_dir, replicate=1, input_transform=None, target_transform=None, RGB=True, noise_level=0.0):
        """Init with directory, transforms and RGB switch."""
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if _is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.replicate = replicate
        self.classes = [None]
        self.RGB = RGB
        self.noise_level = noise_level

    def __getitem__(self, index):
        """Index into dataset."""
        input = _load_img(self.image_filenames[index % len(self.image_filenames)], RGB=self.RGB)
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.noise_level > 0:
            # Add noise
            input += self.noise_level * torch.randn_like(input)

        return input, target

    def __len__(self):
        """Length is amount of files found."""
        return len(self.image_filenames) * self.replicate

class FFHQFolder(ImageFolder):
    
    def __init__(self, root, transform=None, target_transform=None ):
        super(FFHQFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, class_idx = self.samples[index]
        sample = self.loader(path)

        image_num = basename(path).split(".")[0]
        json_path = join(self.root, "../json", f"{image_num}.json")
        # print(json_path)

        with open(json_path) as json_file:
            json_data = json.load(json_file)
        try:
            target = int(json_data[0]["faceAttributes"]["age"] // 10)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            sample, target = self.__getitem__(index+1)

        return sample, target

        

