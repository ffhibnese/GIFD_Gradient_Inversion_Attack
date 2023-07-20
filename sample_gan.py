import pickle
import os

import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

filename = 'results/G_b6dfe5d78698627ab719fd4a6f74aae2.pkl'
out_filename = 'sample_gan_result_c4/0.png'
out_dir = 'sample_gan_result_c4/200/'

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

with open(filename, 'rb') as f:
    G, latents = pickle.load(f)
    G = G.to(device)
    latents = torch.cat(latents).to(device)

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
dm = torch.Tensor(cifar10_mean).view(3, 1, 1).expand(3, 32, 32)
ds = torch.Tensor(cifar10_std).view(3, 1, 1).expand(3, 32, 32)

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


z = torch.randn(16, 100, 1, 1).to(device)
x = G(z)
x = (x + 1) / 2

# z = torch.randn([16, G.z_dim]).to(device)
# z = G.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)

# w = torch.randn_like(z).to(device)
# x = G.synthesis(w, noise_mode='const', force_fp32=True)
# x = (x + 1) / 2


# z = torch.randn([16, G.z_dim]).to(device)
# x = G(z, None, truncation_psi=0.5, truncation_cutoff=8)
# x = (x + 1) / 2

# mean_w = torch.mean(torch.cat(latents), dim=0)
# w = torch.randn([16, 8, 512]).to(device) + mean_w.to(device).unsqueeze(0).expand(16, 8, 512)
# w = latents[-16:]
# x = G.synthesis(w, noise_mode='const', force_fp32=True)
# x = (x + 1) / 2


# fig = plt.figure(figsize=(6, 6))

# for i in range(4):
#     for j in range(4):
#         plt.subplot(6, 6, (i + 1) * 6 + (j + 1))
#         img = x.detach().permute(0, 2, 3, 1).cpu().clamp(0, 1)[i * 4 + j]
#         plt.imshow(img)
#         plt.axis('off')


# if os.path.isfile(out_filename):
#     os.remove(out_filename)
# plt.savefig(out_filename)
# plt.close(fig)

for idx in range(16):
    img = torch.clamp(x[idx:idx+1, ...], 0,1)
    torchvision.utils.save_image(img, out_dir + f'{idx}.png')
