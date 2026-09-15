"""
Defense methods.

Including:
- Additive noise
- Gradient clipping
- Gradient compression
- Representation perturbation
- Orthogonal gradient sampling (CENSOR)
- Adversarial training sample synthesis (ATSPrivacy)

Additive noise, clipping, compression and representation perturbation are
adapted from https://github.com/zhuohangli/GGL.

CENSOR ("Defense against Gradient Inversion via Orthogonal Subspace Bayesian
Sampling", NDSS 2025) is adapted from its MIT-licensed official implementation
https://github.com/KaiyuanZh/censor. It samples gradients orthogonal to the
true one and keeps the candidate that least increases the training loss.

ATSPrivacy ("Privacy Threats Against Federated Learning ...", ICCV 2021) is
adapted from its MIT-licensed official implementation
https://github.com/gaow0007/ATSPrivacy. Unlike the other defenses it acts on
the input image rather than on the gradient: the client trains on samples that
were transformed by a searched augmentation policy.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps

def additive_noise(input_gradient, std=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), std*torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0 # np.inf
    device = input_gradient[0].device
    grad_tensor = [g.clone().cpu().detach() for g in input_gradient]
    
    if norm_type == np.inf:
        norms = [g.abs().max().to(device) for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    # else:
    #     total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in input_gradient]), norm_type)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grad_tensor]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    gradient = [g.mul_(clip_coef_clamped.to(device)) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    device = input_gradient[0].device
    gradient = [None]*len(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].clone().cpu().detach().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)    # percentile threshold
        grad_tensor = torch.where(abs(input_gradient[i]) < thresh, 0, input_gradient[i])   # zero out the entries below it
        gradient[i] = torch.Tensor(grad_tensor).to(device)
    return gradient


def perturb_representation(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device
    
    gt_data = ground_truth.clone()
    gt_data.requires_grad=True

    # register forward hook to get intermediate layer output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]
        return hook

    # for ResNet-18
    handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)
    
    feature_graph = activation['flatten']

    
    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:,f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/((feature_graph.data[:,f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:,f] = 0
        
    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    
    print('Soteria mask: ', sum(mask))

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)
    
    handle.remove()
    
    return gradient


# --------------------------------------------------------------------------- #
# CENSOR: orthogonal subspace Bayesian sampling
# Adapted from https://github.com/KaiyuanZh/censor (MIT license).
# --------------------------------------------------------------------------- #

def generate_orthogonal_gradient(input_gradient):
    """
    Generate a new gradient that is orthogonal to the input_gradient, layer by
    layer, following Eq. (12) of the CENSOR paper:

        g_lo = g_r - proj_{g_l}(g_r) = g_r - (<g_r, g_l> / <g_l, g_l>) * g_l

    Args:
    - input_gradient: A list of gradients (tensors) for each layer of the model.

    Returns:
    - orthogonal_gradient: A list of gradients (tensors) that are orthogonal to the input_gradient.
    """
    orthogonal_gradient = []

    for grad in input_gradient:
        # Generate a random gradient with the same shape
        random_grad = torch.randn_like(grad)
        random_grad = random_grad / torch.norm(random_grad)

        # Flatten the gradients to treat them as vectors
        grad_flat = grad.flatten()
        random_grad_flat = random_grad.flatten()

        # Subtract the projection of random_grad onto grad (Gram-Schmidt)
        proj_scalar = torch.dot(random_grad_flat, grad_flat) / torch.dot(grad_flat, grad_flat)
        proj_vector = proj_scalar * grad_flat

        # Reshape back to the original shape and append to the list
        orthogonal_grad = (random_grad_flat - proj_vector).view_as(grad)
        orthogonal_gradient.append(orthogonal_grad)

    return orthogonal_gradient


def normalize_orthogonal_gradient(ortho_gradient, original_gradient, fixed_norm=None):
    normalized_gradient = []
    for og, orig_g in zip(ortho_gradient, original_gradient):
        norm_og = torch.norm(og.flatten())
        norm_orig_g = fixed_norm if fixed_norm is not None else torch.norm(orig_g.flatten())
        normalized_g = (og / norm_og) * norm_orig_g if norm_og > 0 else og
        normalized_gradient.append(normalized_g)
    return normalized_gradient


def orthogonal_gradient(input_gradient, model, ground_truth, labels, trials=20,
                        epsilon=1e-4, best_loss=float('inf')):
    """
    CENSOR, Algorithm 1 of the paper.

    Sample T gradients orthogonal to the true one (Phase 1, Eq. 12), normalize
    them layer-wise to the scale of the original gradient (Phase 2) and keep the
    candidate whose one-step update least increases the training loss (Phase 3).
    The best gradient is initialized with the original one (line 11), so the
    defense never returns something worse than releasing the true gradient.

    `epsilon` is the learning rate eta used by the one-step update of line 15,
    i.e. the learning rate of the FL client. `best_loss` is the loss of the
    clean gradient (line 10).

    Returns the (possibly replaced) gradient and the best loss that was found.
    """
    criterion = nn.CrossEntropyLoss()
    # Algorithm 1, line 11: G* = G0
    best_gradient = [g.detach().clone() for g in input_gradient]
    best_loss = float(best_loss)

    original_params = [param.detach().clone() for param in model.parameters()]
    was_training = model.training

    for trial in range(trials):
        noisy_gradient = generate_orthogonal_gradient(input_gradient)
        noisy_gradient = normalize_orthogonal_gradient(noisy_gradient, input_gradient)

        with torch.no_grad():
            for param, original, noise_grad in zip(model.parameters(), original_params, noisy_gradient):
                param.data.copy_(original - noise_grad * epsilon)

            model.eval()
            new_loss = criterion(model(ground_truth), labels)

            if trial % 5 == 0:
                print(f'CENSOR trial {trial}: loss {new_loss.item():.4f}')

            if new_loss < best_loss:
                best_loss = new_loss.item()
                best_gradient = [g.detach().clone() for g in noisy_gradient]

    # Restore the original parameters and the training mode
    with torch.no_grad():
        for param, original in zip(model.parameters(), original_params):
            param.data.copy_(original)
    model.train(was_training)

    input_gradient = [g.clone() for g in best_gradient]

    return input_gradient, best_loss


# --------------------------------------------------------------------------- #
# ATSPrivacy: adversarial training sample synthesis
# Adapted from https://github.com/gaow0007/ATSPrivacy (MIT license).
# --------------------------------------------------------------------------- #

class SubPolicy(object):
    """One function of the AutoAugment augmentation library.

    Adapted from the `autoaugment.py` of the ATSPrivacy repository. The library
    contains 50 of these functions; a transformation policy is a combination of
    at most k = 3 of them (Section IV-D of the paper, search space
    sum_{i=1..3} 50^i = 127,550).

    `p1` is kept because the released policy list carries it, but neither the
    paper nor the released implementation gates the function on it: a policy
    that was selected for a sample is applied with its full magnitude.
    """

    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(0, 0, 0)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.5, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.5, 10),
            "brightness": np.linspace(0.0, 0.5, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, m: img.transform(
                img.size, Image.AFFINE, (1, m * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, m: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, m * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, m: img.transform(
                img.size, Image.AFFINE, (1, 0, m * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, m: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, m * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": rotate_with_fill,
            "color": lambda img, m: ImageEnhance.Color(img).enhance(1 + m * random.choice([-1, 1])),
            "posterize": lambda img, m: ImageOps.posterize(img, int(m)),
            "solarize": lambda img, m: ImageOps.solarize(img, int(m)),
            "contrast": lambda img, m: ImageEnhance.Contrast(img).enhance(1 + m * random.choice([-1, 1])),
            "sharpness": lambda img, m: ImageEnhance.Sharpness(img).enhance(1 + m * random.choice([-1, 1])),
            "brightness": lambda img, m: ImageEnhance.Brightness(img).enhance(1 + m * random.choice([-1, 1])),
            "autocontrast": lambda img, m: ImageOps.autocontrast(img),
            "equalize": lambda img, m: ImageOps.equalize(img),
            "invert": lambda img, m: ImageOps.invert(img),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

    def __call__(self, img):
        # The released implementation applies the function unconditionally.
        img = self.operation1(img, self.magnitude1)
        return img


# The pool of sub-policies released as `policy.py` by the ATSPrivacy authors.
# Their search procedure picks a combination out of a much larger library; we
# expose the released pool and sample from it.
ATS_PRIVACY_POLICIES = [
    SubPolicy(0.1, "invert", 7), SubPolicy(0.2, "contrast", 6), SubPolicy(0.7, "rotate", 2),
    SubPolicy(0.3, "translateX", 9), SubPolicy(0.8, "sharpness", 1),
    SubPolicy(0.9, "sharpness", 3), SubPolicy(0.5, "shearY", 2), SubPolicy(0.7, "translateY", 2),
    SubPolicy(0.5, "autocontrast", 5), SubPolicy(0.9, "equalize", 2),
    SubPolicy(0.2, "shearY", 5), SubPolicy(0.3, "posterize", 5), SubPolicy(0.4, "color", 3),
    SubPolicy(0.6, "brightness", 5), SubPolicy(0.3, "sharpness", 9),
    SubPolicy(0.7, "brightness", 9), SubPolicy(0.6, "equalize", 5), SubPolicy(0.5, "equalize", 1),
    SubPolicy(0.6, "contrast", 7), SubPolicy(0.6, "sharpness", 5),
    SubPolicy(0.7, "color", 5), SubPolicy(0.5, "translateX", 5), SubPolicy(0.3, "equalize", 7),
    SubPolicy(0.4, "autocontrast", 8), SubPolicy(0.4, "translateY", 3),
    SubPolicy(0.2, "sharpness", 6), SubPolicy(0.9, "brightness", 6), SubPolicy(0.2, "color", 8),
    SubPolicy(0.5, "solarize", 0), SubPolicy(0.0, "invert", 0),
    SubPolicy(0.2, "equalize", 0), SubPolicy(0.6, "autocontrast", 0), SubPolicy(0.2, "equalize", 8),
    SubPolicy(0.6, "equalize", 4), SubPolicy(0.9, "color", 5),
    SubPolicy(0.6, "equalize", 5), SubPolicy(0.8, "autocontrast", 4), SubPolicy(0.2, "solarize", 4),
    SubPolicy(0.1, "brightness", 3), SubPolicy(0.7, "color", 0),
    SubPolicy(0.4, "solarize", 1), SubPolicy(0.9, "autocontrast", 0), SubPolicy(0.9, "translateY", 3),
    SubPolicy(0.7, "translateY", 3), SubPolicy(0.9, "autocontrast", 1),
    SubPolicy(0.8, "solarize", 1), SubPolicy(0.8, "equalize", 5), SubPolicy(0.1, "invert", 0),
    SubPolicy(0.7, "translateY", 3), SubPolicy(0.9, "autocontrast", 1),
]


def ats_privacy(inputs, mean_std, num_policies=3, policy_pool=None):
    """
    ATSPrivacy. Unlike the other defenses this one transforms the private image
    before the gradient is computed: the client is assumed to train on samples
    that went through a searched augmentation policy, which makes the released
    gradient harder to invert.

    `inputs` is a normalized BCHW tensor and `mean_std` the (mean, std) pair it
    was normalized with, both shaped [C, 1, 1]. Following Section IV-E of the
    paper, a policy is drawn per sample and consists of `num_policies` distinct
    functions of the pool, with k = 3 as in the paper.
    """
    dm, ds = mean_std
    pool = ATS_PRIVACY_POLICIES if policy_pool is None else policy_pool

    defended = []
    for img in inputs:
        pil = transforms.ToPILImage()(torch.clamp(img * ds + dm, 0, 1).cpu())
        for sub_policy in random.sample(pool, min(num_policies, len(pool))):
            pil = sub_policy(pil)
        tensor = transforms.ToTensor()(pil).to(img.device)
        defended.append((tensor - dm) / ds)

    return torch.stack(defended)