"""Measure the cost of one attack iteration for a given FL setup.

The script builds the global model with the same helper used by `rec_mult.py`
and reports, for one forward+backward pass:

- wall-clock time per iteration (median over `--iters` runs)
- forward FLOPs, counted from the Conv2d / Linear layers with forward hooks
- peak GPU memory
- energy, when `pynvml` is installed and a NVIDIA GPU is used

FLOPs are reported for the forward pass only; the backward pass of a
convolution or a linear layer costs roughly twice as much again.

Example
-------
    python tools/profile_cost.py --model ResNet18 --resolution 64 --batch-size 2
    python tools/profile_cost.py --model ViT-Base --resolution 224 --batch-size 2
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inversefed.nn.models import construct_model

# Number of multiply-adds per output element of a Conv2d / Linear layer.
FLOP_HOOKS = {}


def _conv_flops(module, inp, out):
    if isinstance(inp, (list, tuple)):
        inp = inp[0]
    _, cin, h, w = inp.shape
    cout, _, kh, kw = module.weight.shape
    groups = module.groups
    # 2 FLOPs per multiply-add
    n = 2 * out.numel() * (cin // groups) * kh * kw
    if module.bias is not None:
        n += out.numel()
    FLOP_HOOKS['forward'] += n


def _linear_flops(module, inp, out):
    if isinstance(inp, (list, tuple)):
        inp = inp[0]
    n = 2 * inp.numel() * out.shape[-1]
    if module.bias is not None:
        n += out.numel()
    FLOP_HOOKS['forward'] += n


def count_forward_flops(model, x):
    FLOP_HOOKS['forward'] = 0
    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(_conv_flops))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(_linear_flops))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return FLOP_HOOKS['forward']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, seed = construct_model(args.model, num_classes=args.num_classes, num_channels=3,
                                  seed=0, image_size=args.resolution)
    model.to(device).train()
    print(f'model={args.model}  seed={seed}  device={device}')

    x = torch.randn(args.batch_size, 3, args.resolution, args.resolution, device=device)
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step():
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), labels)
        loss.backward()

    for _ in range(args.warmup):
        step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.iters):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else None
    flops = count_forward_flops(model, x)

    print()
    print(f'resolution      : {args.resolution} x {args.resolution}')
    print(f'batch size      : {args.batch_size}')
    print(f'time / iter (s) : {statistics.median(times):.4f}')
    print(f'forward GFLOPs  : {flops / 1e9:.2f}')
    print(f'peak mem (MB)   : {peak_mem_mb:.1f}' if peak_mem_mb is not None
          else 'peak mem (MB)   : n/a (CUDA only)')

    if device.type != 'cuda':
        print('energy / iter(J): n/a (CUDA only)')
        return

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        t0 = time.perf_counter()
        power = []
        for _ in range(args.iters):
            step()
            power.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        energy = statistics.mean(power) * (time.perf_counter() - t0) / args.iters
        print(f'energy / iter(J): {energy:.2f}')
    except Exception as exc:  # pynvml missing, no GPU, or no permission
        print(f'energy / iter(J): n/a ({type(exc).__name__})')


if __name__ == '__main__':
    main()
