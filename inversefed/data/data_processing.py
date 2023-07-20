"""Repeatable code parts concerning data loading."""


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

from ..consts import *

from .data import _build_bsds_sr, _build_bsds_dn
from .loss import Classification, PSNR
from .datasets import FFHQFolder


resize_dict = {
    'ImageNet': 256, 'ImageNet_io' : 32,
    'I256': 256, 'I128': 144, 'I64': 72, 'I32': 36,
    'C10':32, 'C100':32, 
    'PERM':64
}
centercrop_dict = {
    'ImageNet': 224, 'ImageNet_io' : 32,
    'I256': 256, 'I128': 128, 'I64': 64, 'I32': 32,
    'C10':32, 'C100':32,
    'PERM':64
}

def construct_dataloaders(dataset, defs, data_path='~/data', shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'IMAGENET_IO': 
        trainset = [None]
        validset = _build_imagenet_io(path, defs.augmentations, normalize, size=64)
        loss_fn = Classification()
    elif dataset.startswith('I'):
        trainset, validset = _build_imagenet(path, defs.augmentations, normalize, dataset=dataset)
        loss_fn = Classification()
    elif dataset == 'PERM':
        trainset, validset = _build_permuted_Imagenet(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'FFHQ':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'FFHQ64':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=64)
        loss_fn = Classification()
    elif dataset == 'FFHQ128':
        trainset, validset = _build_FFHQ(path, defs.augmentations, normalize, size=128)
        loss_fn = Classification()
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, defs.augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'OOD_FFHQ': 
        trainset = [None]
        validset = _build_ood_ffhq(path, defs.augmentations, normalize, size=64)
        loss_fn = Classification()
    elif dataset == 'OOD_IMAGENET': 
        trainset = [None]
        validset = _build_ood_imagenet(path, defs.augmentations, normalize, size=64)
        loss_fn = Classification()



    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),     
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return loss_fn, trainloader, validloader


def _build_cifar10(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True, dataset='I128'):
    """Define ImageNet with everything considered."""
    # Load data
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(resize_dict[dataset]),
        transforms.CenterCrop(centercrop_dict[dataset]),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset



def _build_ood_ffhq(data_path, augmentations=True, normalize=True, size=64):
    
    # Load data
    #TODO:
    data_mean, data_std = cifar10_mean, cifar10_std
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    validset = ImageFolder(root=data_path, transform=transform)

    return validset

def _build_imagenet_io(data_path, augmentations=True, normalize=True, size=32):
        # Load data
    data_mean, data_std = imagenet_io_mean, imagenet_io_std
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    validset = ImageFolder(root=data_path, transform=transform)

    return validset

def _build_ood_imagenet(data_path, augmentations=True, normalize=True, size=32):
        # Load data
    data_mean, data_std = imagenet_io_mean, imagenet_io_std
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    validset = ImageFolder(root=data_path, transform=transform)

    return validset

def _build_FFHQ(data_path, augmentations=True, normalize=True, size=32):
    """Define ImageNet with everything considered."""
    # Load data
    data_mean, data_std = cifar10_mean, cifar10_std
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    full_set = FFHQFolder(root=data_path, transform=transform)
    #I'm afraid that the FFHQ trainset's size should be 60000, so I correct it from 10000 to 60000.
    trainset = torch.utils.data.Subset(full_set, range(60000))
    validset = torch.utils.data.Subset(full_set, range(len(full_set)))

    trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_permuted_Imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    data_mean, data_std = i64_mean, i64_std

    size=64
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    
    full_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    trainset = full_set
    validset = full_set

    trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _get_meanstd(dataset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std
