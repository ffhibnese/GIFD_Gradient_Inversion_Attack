"""Train the classifier that acts as the semantic remapper of the label mapping technique.

Under label inconsistency the label inferred from the FL gradients belongs to
the *private* label space, while a conditional generator expects the label space
it was trained on. Label mapping therefore inverts twice: a coarse pass produces
a rough reconstruction, a classifier f_m(.) trained on the GAN's own dataset
remaps it to a generator-compatible label, and a second fine-grained pass uses
that label as conditioning.

This script trains f_m(.). Two things matter:

- f_m has to be trained on the dataset the GAN was trained on, so that its
  output really is a label of the generator's label space.
- the images must be normalized exactly like the private data during the
  attack, otherwise f_m sees a different input distribution than the coarse
  reconstruction it is applied to.

Example
-------
    python tools/train_label_mapper.py \
        --data ./dataset/media/imagenet/train \
        --val-data ./dataset/media/imagenet/val \
        --dataset IMAGENET_IO \
        --model ResNet18 --resolution 64 --epochs 20 \
        --out label_mapper.pt
"""

import argparse
import os
import sys

import torch
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inversefed
from inversefed import consts
from inversefed.data.loss import Classification

# Datasets that reuse the CIFAR statistics, mirroring rec_mult.py.
CIFAR_STATS_DATASETS = ('FFHQ',)


def _mean_std(dataset):
    if dataset.startswith(CIFAR_STATS_DATASETS) or dataset.endswith(CIFAR_STATS_DATASETS):
        key = 'cifar10'
    else:
        key = dataset.lower()
    return (getattr(consts, f'{key}_mean'), getattr(consts, f'{key}_std'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='ImageFolder with the GAN training data')
    parser.add_argument('--val-data', default=None,
                        help='optional ImageFolder used for validation; '
                             'if omitted the training set is split 90/10')
    parser.add_argument('--dataset', default='IMAGENET_IO',
                        help='dataset key whose normalization is applied (see inversefed.consts)')
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='label_mapper.pt')
    args = parser.parse_args()

    setup = inversefed.utils.system_startup(args)
    mean, std = _mean_std(args.dataset)

    train_transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = datasets.ImageFolder(args.data, transform=train_transform)
    num_classes = len(trainset.classes)
    print(f'{len(trainset)} training images, {num_classes} classes')

    if args.val_data:
        validset = datasets.ImageFolder(args.val_data, transform=eval_transform)
    else:
        valid_size = max(1, len(trainset) // 10)
        trainset, validset = random_split(trainset, [len(trainset) - valid_size, valid_size])
        trainset.dataset.transform = train_transform
        validset.dataset.transform = eval_transform

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              drop_last=True, num_workers=args.workers,
                                              pin_memory=inversefed.consts.PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False,
                                              drop_last=False, num_workers=args.workers,
                                              pin_memory=inversefed.consts.PIN_MEMORY)

    model, model_seed = inversefed.construct_model(args.model, num_classes=num_classes,
                                                   num_channels=3, seed=0,
                                                   image_size=args.resolution)
    model.to(**setup)

    defs = inversefed.training_strategy('conservative')
    defs.epochs = args.epochs
    defs.batch_size = args.batch_size
    defs.lr = args.lr

    loss_fn = Classification()

    print(f'Training {args.model} for {args.epochs} epochs ...')
    inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)

    torch.save({'arch': args.model,
                'num_classes': num_classes,
                'resolution': args.resolution,
                'dataset': args.dataset,
                'mean': mean,
                'std': std,
                'model_seed': model_seed,
                'state_dict': model.state_dict()},
               args.out)
    print(f'Saved the semantic remapper to {args.out}')


if __name__ == '__main__':
    main()
