"""Simulate the federated deployment used by the FL-setting experiments.

Two things are needed to attack a realistic FL system instead of a randomly
initialised model:

1. A non-IID split of the private data among clients. Following the paper, the
   dataset classes are randomly distributed over Nclient clients, and a sample
   with label l goes to its designated client with probability q and to any
   other client with probability (1 - q) / (Nclient - 1). With q = 1 / Nclient
   the local datasets are IID; the paper reports q in
   {0.10, 0.30, 0.50, 0.70, 0.90, 0.99} with Nclient = 10.
2. The global model after a given number of FL rounds, because the attacks are
   run against that model. The paper increases the number of rounds and attacks
   the corresponding global model.

Example
-------
    python tools/fl_simulation.py split \
        --dataset IMAGENET_IO --data-path ./dataset/media/imagenet/train \
        --num-clients 10 --q 0.10 --out fl_split_q0.10.json

    python tools/fl_simulation.py train \
        --dataset IMAGENET_IO --data-path ./dataset/media/imagenet/train \
        --split fl_split_q0.10.json --rounds 5 --local-steps 1 \
        --local-lr 0.1 --model ResNet18 --out-dir fl_rounds

    python rec_mult.py --config configs_biggan.yml \
        --model_ckpt fl_rounds/global_round_5.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inversefed


def split_by_q(targets, num_clients=10, q=0.1, seed=0):
    """Assign every sample to a client.

    The classes are randomly distributed over the clients; a sample then goes to
    its designated client with probability q and to any of the other clients
    with probability (1 - q) / (num_clients - 1). q = 1 / num_clients gives the
    IID split.
    """
    targets = np.asarray(targets)
    labels = np.unique(targets)
    rng = np.random.default_rng(seed)

    designated = {int(l): int(c) for l, c in
                  zip(labels, rng.integers(0, num_clients, size=len(labels)))}

    clients = [[] for _ in range(num_clients)]
    stay = rng.random(len(targets)) < q
    others = rng.integers(0, num_clients - 1, size=len(targets))

    for i, label in enumerate(targets):
        home = designated[int(label)]
        if stay[i] or num_clients == 1:
            clients[home].append(i)
        else:
            # any client but the designated one
            offset = int(others[i])
            clients[(home + 1 + offset) % num_clients].append(i)

    return clients, designated


def _build_dataset(args):
    """The dataset rec_mult samples its targets from, so indices stay valid."""
    defs = inversefed.training_strategy('conservative')
    _, _, validloader = inversefed.construct_dataloaders(
        args.dataset, defs, data_path=args.data_path)
    return validloader.dataset


def _targets_of(dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    if hasattr(dataset, 'labels'):
        return dataset.labels
    # ImageFolder: samples is a list of (path, class)
    if hasattr(dataset, 'samples'):
        return [s[1] for s in dataset.samples]
    raise SystemExit('cannot read the labels of this dataset')


def cmd_split(args):
    dataset = _build_dataset(args)
    targets = _targets_of(dataset)
    clients, designated = split_by_q(targets, args.num_clients, args.q, args.seed)

    payload = {
        'dataset': args.dataset,
        'num_clients': args.num_clients,
        'q': args.q,
        'seed': args.seed,
        'designated_client_by_label': {str(k): v for k, v in designated.items()},
        'clients': clients,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f)
    sizes = [len(c) for c in clients]
    print(f'{len(targets)} samples -> {args.num_clients} clients (q={args.q})')
    print(f'  per-client sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}')
    print(f'  written to {args.out}')


def cmd_train(args):
    with open(args.split) as f:
        split = json.load(f)
    clients = split['clients']
    dataset = _build_dataset(args)

    setup = inversefed.utils.system_startup(args)
    model, _ = inversefed.construct_model(args.model, num_classes=args.num_classes,
                                          num_channels=3, seed=args.seed,
                                          image_size=args.resolution)
    model.to(**setup)

    loss_fn = torch.nn.CrossEntropyLoss()
    os.makedirs(args.out_dir, exist_ok=True)

    for rnd in range(1, args.rounds + 1):
        global_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        total = 0
        delta = None

        for cid, idx in enumerate(clients):
            if not idx:
                continue
            model.load_state_dict(global_sd)
            model.train()
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, idx), batch_size=args.batch_size,
                shuffle=True, drop_last=False, num_workers=args.workers)

            opt = torch.optim.SGD(model.parameters(), lr=args.local_lr,
                                  momentum=0.9, weight_decay=args.weight_decay)
            for _ in range(args.local_steps):
                for x, y in loader:
                    x, y = x.to(**setup), y.to(**setup)
                    opt.zero_grad()
                    loss_fn(model(x), y).backward()
                    opt.step()

            n = len(idx)
            step = {k: (v.detach().float() - global_sd[k].float()) * n
                    for k, v in model.state_dict().items()}
            if delta is None:
                delta = step
            else:
                for k in delta:
                    delta[k] += step[k]
            total += n
            print(f'  round {rnd}: client {cid} done ({n} samples)')

        new_sd = {k: global_sd[k].float() + delta[k] / total for k in global_sd}
        model.load_state_dict({k: v for k, v in new_sd.items()})
        path = os.path.join(args.out_dir, f'global_round_{rnd}.pt')
        torch.save({'round': rnd, 'arch': args.model, 'num_classes': args.num_classes,
                    'state_dict': model.state_dict()}, path)
        print(f'  saved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('split', help='assign the private samples to clients')
    p.add_argument('--dataset', default='IMAGENET_IO')
    p.add_argument('--data-path', required=True)
    p.add_argument('--num-clients', type=int, default=10)
    p.add_argument('--q', type=float, default=0.1,
                   help='probability of staying on the designated client; '
                        '1 / num_clients gives the IID split')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='fl_split.json')
    p.set_defaults(func=cmd_split)

    p = sub.add_parser('train', help='run FedAvg and store the global model per round')
    p.add_argument('--dataset', default='IMAGENET_IO')
    p.add_argument('--data-path', required=True)
    p.add_argument('--split', required=True, help='json written by the split command')
    p.add_argument('--model', default='ResNet18')
    p.add_argument('--num-classes', type=int, default=1000)
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--local-steps', type=int, default=1)
    p.add_argument('--local-lr', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='fl_rounds')
    p.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
