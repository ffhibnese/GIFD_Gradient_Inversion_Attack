# GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A PyTorch official implementation for [GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization](), accepted to ICCV-2023.

![pipeline](./figures/pipeline.png)

## Method

Instead of searching only the initial latent code of a pre-trained GAN, GIFD **disassembles
the generator** into `G0 o G1 o ... o GN` and progressively changes the layer being
optimized, from the initial latent space to intermediate layers closer to the output image.
Because the intermediate feature space is much larger and may lead to unrealistic image
generation, the searching range of each layer is restricted to a small **l1 ball** centered
at the vector induced by the previous layer. The output of the layer attaining the least
gradient matching loss is selected as the final reconstruction. The per-layer radii are
configured through `max_radius_*` in the yml files.

## Results
![results](./figures/results.jpg)

## Setup

We provide the environment configuration file exported by Anaconda, which can help you build up conveniently.
```bash
conda env create -f environment.yml
conda activate GIFD
```

For newer software stacks we also provide a pip-based `requirements.txt` with lower bounds
only (`pip install -r requirements.txt`).

> Note: `inversefed/genmodels/stylegan2_io/op` JIT-compiles two CUDA extensions
> (`fused_bias_act`, `upfirdn2d`) at *import* time, so a working `nvcc` matching your
> installed CUDA toolkit is required.

## Dataset and model file

Download the [ImageNet](https://www.image-net.org/) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) and provide their paths in the yml file.

While the model weights of BigGAN are downloaded automatically, StyleGAN2 weights require downloading manually as follows.

```bash
# shape predictor / W+ space Gaussian fit -> place in the root directory as gaussian_fit.pt
gdown --id 1c1qtz3MVTAvJpYvsMIR5MoSvdiwN2DGb

# stylegan pre-trained checkpoint -> place in inversefed/genmodels/stylegan2_io/
gdown --id 1JCBiKY_yUixTa6F1eflABL88T4cii2GR
```

The optional StyleGAN2-ADA decoders are not shipped with the repository; if you need them,
point the `STYLEGAN2_ADA_CKPT` environment variable at a local `network-snapshot-*.pkl`.

## Quick start

We prepare three configuration files for performing gradient inversion attacks, including the BigGAN-based, the StyleGAN2-based, and the GAN-free methods, where we give detailed descriptions of every parameter. Feel free to contact me at fang-h23@mails.tsinghua.edu.cn if you have any concerns.

```bash
python rec_mult.py --config configs_biggan.yml      # ImageNet 64x64, BigGAN prior
python rec_mult.py --config configs_stylegan2.yml   # FFHQ 64x64, StyleGAN2 prior
python rec_mult.py --config configs_gan_free.yml    # Geiping / Yin baselines, no GAN
```

Metrics (PSNR, LPIPS-VGG, LPIPS-Alex, SSIM, MSE) are appended to
`<output_dir>/<exp_name>/table_Metrics.csv` and the reconstructed images are saved next to it.

Defenses are configured per experiment through `defense_method` / `defense_setting`
(`noise`, `clipping`, `compression`, `representation`), implemented in `defense.py`.

## Citation
```
@inproceedings{fang2023gifd,
  title={GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization},
  author={Fang, Hao and Chen, Bin and Wang, Xuan and Wang, Zhi and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4967--4976},
  year={2023}
}
```

## Acknowledgement
Our code is based on [invertingGradients](https://github.com/JonasGeiping/invertinggradients) and [ILO](https://github.com/giannisdaras/ilo) and we are grateful for their great devotion.

For BigGAN, we use the PyTorch [implementation and weights](https://github.com/huggingface/pytorch-pretrained-BigGAN).

For StyleGAN2, we adapt this [Pytorch implementation](https://github.com/rosinality/stylegan2-pytorch), which is based on the [official Tensorflow code](https://github.com/NVlabs/stylegan2).

We express great gratitude for their contribution to our community!
