"""Various utilities."""

import os
import csv

import torch
import random
import numpy as np

import socket
import datetime
import pandas as pd
from mae import models_mae
from matplotlib import pyplot as plt
from .nn import inn
import pickle

def system_startup(args=None, defs=None):
    """Print useful system information."""
    # Choose GPU device and print status information:
    
    #!!!!!!!!!!!!!!!!!!!
    #pay attention
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    # if args is not None:
        # print(args)
    if defs is not None:
        print(repr(defs))
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return setup

def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        if os.path.isfile(fname):
            old_table = pd.read_csv(fname)
        data = old_table.append(kwargs, ignore_index=True)
        # with open(fname, 'r') as f:
        #     reader = csv.reader(f, delimiter='\t')
        #     header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        data = pd.DataFrame(columns = fieldnames)
        data = data.append(kwargs, ignore_index=True)
    if not dryrun:
        # Add row for this experiment
        data.to_csv(fname, index=None)
    else:
        print(f'Would save results to {fname}.')
        print(f'Would save these keys: {fieldnames}.')

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    imagenet_std = [0.485, 0.456, 0.406]
    imagenet_mean = [0.229, 0.224, 0.225]
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_inn(latent_dim, fpath):

    base_to_latent = inn.create_inn(
        latent_dim,
        8,
        "all_in_one",
        )
    if fpath:
        base_to_latent.load_state_dict(torch.load(fpath))        # inn.load_state_dict(torch.load('scg.pth')))
        
    return base_to_latent
