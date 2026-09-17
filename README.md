# GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A PyTorch official implementation for [GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization](), accepted to ICCV-2023.

![pipeline](./figures/pipeline.png)

## Contents

- [Results](#results)
- [Setup](#setup)
- [Dataset and model files](#dataset-and-model-files)
- [Supported FL models and datasets](#supported-fl-models-and-datasets)
- [Quick start](#quick-start)
- [Method](#method)
  - [Optimization schedule](#optimization-schedule)
  - [Ablation variants](#ablation-variants)
  - [Label mapping](#label-mapping)
  - [Federated learning settings](#federated-learning-settings)
- [Tools](#tools)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Results
![results](./figures/results.jpg)

## Setup

Install the environment with pip (a conda environment works just as well):

```bash
pip install -r requirements.txt
```

`opencv-python` and `pillow` are used by the data pipelines, `lpips` by the
evaluation, and `nevergrad` by the CMA-ES baseline. `timm` is only needed for the
DeiT global models and `gdown` only to download the StyleGAN2 weights, so both
can be skipped if you do not use them.

> Note: `inversefed/genmodels/stylegan2_io/op` JIT-compiles two CUDA extensions
> (`fused_bias_act`, `upfirdn2d`) at *import* time. This happens before anything
> else, so a single missing or unsuitable `nvcc` makes the whole StyleGAN2 path
> fail to import. The extensions are built with `-std=c++17`, which means
> **CUDA 11 or newer is required**; with older toolkits the build stops at
> `nvcc fatal : Value 'c++17' is not defined for option 'std'`. On machines whose
> default `nvcc` is old, point the environment at a newer toolkit first, e.g.
>
> ```bash
> export CUDA_HOME=/usr/local/cuda-11.8
> export PATH=$CUDA_HOME/bin:$PATH
> ```
>
> The BigGAN path is unaffected, as it uses no custom operators.

## Dataset and model files

Download the [ImageNet](https://www.image-net.org/) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) and provide their paths in the yml file.

While the model weights of BigGAN are downloaded automatically, StyleGAN2 weights require downloading manually as follows.

```bash
# pre-trained StyleGAN2 (FFHQ) checkpoint -> place in inversefed/genmodels/stylegan2_io/
gdown --id 1JCBiKY_yUixTa6F1eflABL88T4cii2GR
```

The StyleGAN2 path additionally needs `gaussian_fit.pt` in the repository root,
the Gaussian fit of the W+ space used by `MappingProxy`. It is a **torch** file
holding two tensors of shape `(512,)`, `mean` and `std`; it is loaded with
`torch.load` at `inversefed/reconstruction_algorithms.py:216`. The CENSOR
repository (https://github.com/KaiyuanZh/censor) ships such a file at its root.

> Note: do not confuse it with the dlib *shape predictor* (~99 MB) that some
> instructions link to for aligning FFHQ faces. That one is a dlib model, not a
> torch file, and nothing in this code base loads it — pointing it at
> `gaussian_fit.pt` makes `torch.load` fail with `UnpicklingError`.

The optional StyleGAN2-ADA decoders are not shipped with the repository; if you need them,
point the `STYLEGAN2_ADA_CKPT` environment variable at a local `network-snapshot-*.pkl`.

## Supported FL models and datasets

The victim (global) model is selected with the `model` field of a config and is built by
`inversefed.nn.models.construct_model`. Besides the architectures inherited from
`invertingGradients` (`ConvNet`, the `ResNet` family, `DenseNet-121`, `MobileNet`,
`MNASNet`, ...), we additionally support:

| `model` | Source | Note |
|---|---|---|
| `AlexNet`, `VGG-16` | torchvision | |
| `ViT-Base`, `ViT-Large`, `ViT-Huge` | torchvision | patch size 16 / 16 / 14 |
| `DeiT-Small`, `DeiT-Tiny` | `timm` | install with `pip install timm` |

Transformer architectures build their positional embeddings from the input resolution, so
the image size of the FL dataset is passed to the constructor automatically. Note that the
image size must be a multiple of the patch size (e.g. `ViT-Huge` uses patches of 14 and
therefore does not accept 64 x 64 inputs).

Private datasets are selected with the `dataset` field and are built by
`inversefed.data.construct_dataloaders`: `IMAGENET_IO` (ImageNet ILSVRC 2012), `FFHQ64` /
`FFHQ128`, `CIFAR10`, `CIFAR100`, `MNIST`, `KMNIST`, `SVHN`, and the out-of-distribution
variants `OOD_FFHQ` / `OOD_IMAGENET`.

## Quick start

We prepare three configuration files for performing gradient inversion attacks, including the BigGAN-based, the StyleGAN2-based, and the GAN-free methods, where we give detailed descriptions of every parameter. Feel free to contact me at fang-h23@mails.tsinghua.edu.cn if you have any concerns.

```bash
python rec_mult.py --config configs_biggan.yml        # ImageNet 64x64, BigGAN prior
python rec_mult.py --config configs_stylegan2.yml     # FFHQ 64x64, StyleGAN2 prior
python rec_mult.py --config configs_gan_free.yml      # Geiping / Yin baselines, no GAN
python rec_mult.py --config configs_biggan_256.yml    # ImageNet 256x256, DenseNet-121
python rec_mult.py --config configs_ood_biggan.yml    # private data from a shifted domain
python rec_mult.py --config configs_ood_stylegan2.yml # face prior vs. ImageNet data
```

The last three cover the settings that go beyond the default one: higher
resolution, an out-of-distribution private domain, and the extreme case where
the generative prior (faces) shares almost no semantics with the private data
(ImageNet classes).

Metrics (PSNR, LPIPS-VGG, LPIPS-Alex, SSIM, MSE) are appended to
`<output_dir>/<exp_name>/table_Metrics.csv` and the reconstructed images are saved next to it.

### Defenses

Defenses are configured per experiment through `defense_method` / `defense_setting`:

| `defense_method` | Defense | Acts on |
|---|---|---|
| `noise` | additive Gaussian noise (DP) | gradient |
| `clipping` | gradient clipping by norm | gradient |
| `compression` | gradient sparsification | gradient |
| `representation` | Soteria, perturbs the representation | gradient |
| `orthogonal` | CENSOR, orthogonal subspace Bayesian sampling | gradient |
| `ats_privacy` | ATSPrivacy, searched augmentation policy | input image |

CENSOR samples `our_num_tries` gradients orthogonal to the true one and keeps
the candidate whose one-step update least increases the training loss. It is
adapted from the official [implementation](https://github.com/KaiyuanZh/censor)
following Eq. (12) of the paper. ATSPrivacy is adapted from the official
[implementation](https://github.com/gaow0007/ATSPrivacy); unlike the others it
transforms the private image instead of the gradient, so it is applied before
the shared gradient is computed.

## Method

Instead of searching only the initial latent code of a pre-trained GAN, GIFD **disassembles
the generator** into `G0 o G1 o ... o GN` and progressively changes the layer being
optimized, from the initial latent space to intermediate layers closer to the output image.
Because the intermediate feature space is much larger and may lead to unrealistic image
generation, the searching range of each layer is restricted to a small **l1 ball** centered
at the vector induced by the previous layer. The output of the layer attaining the least
gradient matching loss is selected as the final reconstruction. The per-layer radii are
configured through `max_radius_*` in the yml files. Two regularization terms are added to
the matching loss, weighted by `total_variation` (alpha_TV = 1e-4) and `image_norm`
(alpha_l2 = 1e-6) in the configs.

### Optimization schedule

Each search stage runs `steps[k]` Adam iterations starting from `lr_io[k]` (0.1 by
default). The learning rate is not constant: it follows `get_lr` in
`inversefed/reconstruction_algorithms.py`, which linearly warms the rate up from 0 over
the first 1/20 of the iterations (`rampup = 0.05`) and then decays it to 0 with a cosine
profile over the remaining 3/4 of them (`rampdown = 0.75`).

By default every stage restarts that schedule, so each layer gets its own warm-up and
decay. Setting `lr_same_pace: true` instead computes `t` over the total number of
iterations of all stages, so the whole hierarchy shares one warm-up and one decay; this
currently only affects the StyleGAN2 path.

### Ablation variants

`gifd_variant` selects one of the variants of Table VIII, which isolate the
contribution of each technique:

| `gifd_variant` | Searches | l1 ball | Reported output |
|---|---|---|---|
| `z` | latent space only | no | the latent-space reconstruction |
| `f` | latent + intermediate features | no | the **last** searched layer |
| `e` | latent + intermediate features | no | the layer with the **least** matching error |
| `full` (default) | latent + intermediate features | yes | the layer with the least matching error |

`GIFD-e` is `GIFD-f` with the output-selection rule, and `GIFD` is `GIFD-e`
plus the l1 ball limitation. The variant is reported under its own name in
`table_Metrics.csv`, so it can be read off without having to know which of the
`layer*` columns to look at.

### Label mapping

Under label inconsistency the label inferred from the shared gradients belongs to
the *private* label space, while a conditional generator expects the label space
it was trained on: attacking a face classifier with an ImageNet-pretrained
BigGAN, the inferred label `0` means "age 0-9" for the private data but "tench"
for the generator. Feeding that label to the generator provides misleading
conditioning and degrades the reconstruction.

Label mapping inverts twice. A first coarse pass runs only `coarse_iterations`
iterations per layer using the inferred label; the resulting image is then
classified by a remapper `f_m(.)` trained on the GAN's own dataset, whose
prediction is by construction a label of the generator's label space; a second
fine-grained pass uses that label as conditioning. The remapper only needs to be
trained once per generative prior, see [Tools](#tools):

```bash
python tools/train_label_mapper.py \
    --data ./dataset/media/imagenet/train \
    --val-data ./dataset/media/imagenet/val \
    --dataset IMAGENET_IO --model ResNet18 --resolution 64 \
    --epochs 20 --out label_mapper.pt
```

Then point `label_mapper_ckpt` of a config at the checkpoint and set
`label_mapping: true` (`configs_ood_biggan.yml` does this). The technique only
applies to conditional generators, so it stays disabled for the StyleGAN2 prior.

### Federated learning settings

The default configs attack a randomly initialized global model with IID private
data. `tools/fl_simulation.py` covers the two FL factors the experiments also
vary, see [Tools](#tools):

- **Data heterogeneity.** The dataset classes are randomly distributed over
  `Nclient` clients; a sample with label `l` goes to its designated client with
  probability `q` and to any other client with probability
  `(1 - q) / (Nclient - 1)`. With `Nclient = 10`, `q = 0.10` is the IID case and
  larger values give increasingly non-IID clients.
- **Number of FL rounds.** `train` runs FedAvg and stores the global model after
  every round, so an attack can be pointed at a converged model instead of a
  randomly initialized one.

```bash
python tools/fl_simulation.py split --dataset IMAGENET_IO \
    --data-path ./dataset/media/imagenet/val --num-clients 10 --q 0.10 \
    --out fl_split_q0.10.json

python tools/fl_simulation.py train --dataset IMAGENET_IO \
    --data-path ./dataset/media/imagenet/val --split fl_split_q0.10.json \
    --rounds 5 --local-steps 1 --local-lr 0.1 --out-dir fl_rounds

python rec_mult.py --config configs_biggan.yml \
    --model_ckpt fl_rounds/global_round_5.pt \
    --split_file fl_split_q0.10.json --client 0
```

`--model_ckpt` loads the global model of a given round, and `--split_file` with
`--client` restricts the reconstructed images to the private data of one client.
Both commands read the same dataset object that `rec_mult.py` samples its targets
from, so the indices written by `split` stay valid.

## Tools

Four small scripts under `tools/` cover what goes beyond the per-image metrics
written by `rec_mult.py`.

| Script | Purpose | Documented in |
|---|---|---|
| `stat_test.py` | significance of the improvements | below |
| `profile_cost.py` | cost of a single attack iteration | below |
| `fl_simulation.py` | heterogeneous clients and FL rounds | [Federated learning settings](#federated-learning-settings) |
| `train_label_mapper.py` | train the remapper of label mapping | [Label mapping](#label-mapping) |

**Significance of the improvements** (`tools/stat_test.py`). Every attack
appends one row per reconstructed image to its own `table_Metrics.csv`. Point
the script at those tables to obtain the mean and the standard deviation per
method together with a paired t-test of each baseline against a reference
method:

```bash
python tools/stat_test.py \
    --runs GIFD=results/ex1_gifd/table_Metrics.csv \
           GIAS=results/ex1_gias/table_Metrics.csv \
           GGL=results/ex1_ggl/table_Metrics.csv \
    --metric psnr --reference GIFD
```

Runs are paired on the `target_id` column when it is present and on row order
otherwise. The metric column is chosen automatically, preferring the `Best_*`
column that holds the output selected by the least gradient matching loss;
use `--column` to select one explicitly.

**Cost of one attack iteration** (`tools/profile_cost.py`). Builds the global
model with the same helper as `rec_mult.py` and reports the wall-clock time,
the forward FLOPs, the peak GPU memory and (when `pynvml` is available) the
energy of a single forward+backward pass:

```bash
python tools/profile_cost.py --model ResNet18 --resolution 64 --batch-size 2
```

FLOPs are counted for the forward pass only; the backward pass of a
convolution or a linear layer costs roughly twice as much again.

## Citation
```
@inproceedings{fang2023gifd,
  title={GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization},
  author={Fang, Hao and Chen, Bin and Wang, Xuan and Wang, Zhi and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4967--4976},
  year={2023}
}

@article{fang2026enhancing,
  title={Enhancing gradient inversion attacks in federated learning via hierarchical feature optimization},
  author={Fang, Hao and Yu, Wenbo and Chen, Bin and Wang, Xuan and Xia, Shu-Tao and Liao, Qing and Xu, Ke},
  journal={arXiv preprint arXiv:2604.00955},
  year={2026}
}
```

## Acknowledgement
Our code is based on [invertingGradients](https://github.com/JonasGeiping/invertinggradients) and [ILO](https://github.com/giannisdaras/ilo) and we are grateful for their great devotion.

For BigGAN, we use the PyTorch [implementation and weights](https://github.com/huggingface/pytorch-pretrained-BigGAN).

For StyleGAN2, we adapt this [Pytorch implementation](https://github.com/rosinality/stylegan2-pytorch), which is based on the [official Tensorflow code](https://github.com/NVlabs/stylegan2).

We express great gratitude for their contribution to our community!
