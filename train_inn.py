"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torchvision

import numpy as np

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time
import os
import json
import hashlib
import csv
import yaml
import copy
import pickle
import lpips

import inversefed.porting as porting
from inversefed.utils import prepare_inn


nclass_dict = {'I32': 1000, 'I64': 1000, 'I128': 1000, 
               'CIFAR10': 10, 'CIFAR100': 100, 'CA': 8, 'ImageNet':1000,
               'FFHQ': 10, 'FFHQ64': 10, 'FFHQ128': 10,
               'PERM': 1000
               }
# Parse input arguments
parser = inversefed.options()
parser = inversefed.options()
parser.add_argument('--seed', default=1234, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--batch_size', default=4, type=int, help='Number of mini batch for federated averaging')
parser.add_argument('--local_lr', default=1e-4, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--checkpoint_path', default='', type=str, help='Local learning rate for federated averaging')
parser.add_argument('--gan', default='stylegan2', type=str, help='GAN model option:[stylegan2, biggan]')

args = parser.parse_args()


# Parse training strategy
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs


def l2(est_param, target_param):
    param_diff = 0
    for idx, (gx, gy) in enumerate(zip(est_param, target_param)): # TODO: fix the variables here
        if len(gx.shape) >= 1:
            layer_diff = ((gx - gy) ** 2).sum() # / math.sqrt(gx.shape[0])
        else:
            layer_diff = ((gx - gy) ** 2).sum() # / math.sqrt(gx.shape[0])

        param_diff += layer_diff
    return param_diff

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config

if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training
    # if args.gan not in ['stylegan2', 'biggan']:
    #     raise ValueError("The selected gan is not supported!")
    # file_name = args.gan

    config_path = "./train_inn.yml"
    config = load_config(config_path=config_path)
    # Prepare for training
    # Get data:
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(config['dataset'], defs, data_path=config['data_path'])


    if config['dataset'].startswith('FFHQ') or config['dataset'].endswith('FFHQ'):
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    else:
        dataset = config['dataset']
        dm = torch.as_tensor(getattr(inversefed.consts, f'{dataset.lower()}_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'{dataset.lower()}_std'), **setup)[:, None, None]

    model, model_seed = inversefed.construct_model(config['model'], num_classes=nclass_dict[config['dataset']], num_channels=3)
    model.to(**setup)
    model.eval()


    # Load a trained model?
    if args.trained_model:
        file = f'{args.model}_{args.epochs}.pth'
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=setup['device']))
            print(f'Model loaded from file {file}.')
        except FileNotFoundError:
            print('Training the model ...')
            print(repr(defs))
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), os.path.join(args.model_path, file))

    # psnr list
    psnrs = []

    # hash configuration
    config_m = dict(    cost_fn=config['cost_fn'],
                        indices=config['indices'],
                        weights=config['weights'],
                        lr=config['lr'] if config['lr'] is not None else 0.1,
                        optim='adam',
                        restarts=config['restarts'],
                        max_iterations=config['max_iterations'],
                        total_variation=config['total_variation'],
                        bn_stat=config['bn_stat'],
                        image_norm=config['image_norm'],
                        z_norm= config['z_loss'],
                        group_lazy=config['group_lazy'],
                        init=config['init'],
                        lr_decay=True,
                        dataset=config['dataset'],
                        #params for inter optim
                        ckpt= config['ckpt'],                          
                        generative_model=config['generative_model'],
                        gen_dataset=config['gen_dataset'],
                        gias= config['gias'],
                        ggl = config['ggl'],
                        prompt_gen = config['prompt_gen'],
                        prompt = config['prompt'],
                        prompt_iteration = config['prompt_iteration'],   
                        prompt_lr = config['prompt_lr'],
                        inn_path = config['inn_path']
                        )

    # config_hash = hashlib.md5(json.dumps(config_m, sort_keys=True).encode()).hexdigest()



    file_path = os.path.join(config['dataset'], config['exp_name'])
    # os.makedirs('models', exist_ok=True)
    # os.makedirs(f'models/{file_path}', exist_ok=True)


    if args.checkpoint_path:
        with open(args.checkpoint_path, 'rb') as f:
            inn, latents = pickle.load(f)
            inn = inn.requires_grad_(True).to(setup['device'])
    else:
        if config['generative_model'].startswith('stylegan2'):
            latent_dim = 512
        elif config['generative_model'].startswith('BigGAN'):
            latent_dim = 128
        inn = prepare_inn(latent_dim, config['inn_path'])
        inn.to(setup['device'])
        latents = []


    inn_optimizer = torch.optim.Adam(inn.parameters(), lr=config['meta_lr'])

    #lpips
    lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(**setup)
    lpips_loss_a = lpips.LPIPS(net='alex', spatial=False).to(**setup)


    target_id = config['target_id']
    for i in range(config['num_exp']):
        psnrs = {}
        lpips_sc ={}
        lpips_sc_a = {}
        ssim = {}
        mse_i = {}

        target_id = config['target_id'] + i * 100
        tid_list = []
        if config['num_images'] == 1:
            ground_truth, labels = validloader.dataset[target_id]
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
            target_id_ = target_id + 1
            print("loaded img %d" % (target_id_ - 1))
            tid_list.append(target_id_ - 1)
        else:
            ground_truth, labels = [], []
            target_id_ = target_id
            while len(labels) < config['num_images']:
                img, label = validloader.dataset[target_id_]
                target_id_ += 1
                if (label not in labels):
                    print("loaded img %d" % (target_id_ - 1))
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                    ground_truth.append(img.to(**setup))
                    tid_list.append(target_id_ - 1)

            ground_truth = torch.stack(ground_truth)
            labels = torch.cat(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])
        # print("Ground truth's dimension:{}".format(ground_truth.shape))
        # Run reconstruction
        input_gradients = []
        for j in range(config['num_images']):
            model.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth[j].unsqueeze(0)), labels[j].unsqueeze(0))
            input_gradient = torch.autograd.grad(target_loss, model.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]
            input_gradients.append(input_gradient)

        print("Creating Gradient Reconstructor")
        rec_machine = inversefed.GradientReconstructor(model, setup['device'], (dm, ds), config_m, num_images=config['num_images'])
        
        print("Starting Reconstruction")
        result = rec_machine.reconstruct(input_gradients, labels, img_shape=img_shape, dryrun=args.dryrun)
        
        
        inn_optimizer.zero_grad()

        inn_updated = rec_machine.inn
        diff = l2(list(inn_updated.parameters()), list(inn.parameters()))
        diff.backward()
        inn_optimizer.step()

        latents.append(rec_machine.dummy_z)

        if (i + 1) % 50 == 0:
            model.train()
            model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            model_optimizer.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            target_loss.backward()
            model_optimizer.step()
            model.eval()

        # Compute stats and save to a table:
        file_name, output, stats = result[0]

            
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        test_mse = (output_den - ground_truth_den).pow(2).mean().item()
        ssim_score, _ = inversefed.metrics.ssim_batch(output, ground_truth)
        with torch.no_grad():
            lpips_score = lpips_loss(output, ground_truth).squeeze().mean().item()
            lpips_score_a = lpips_loss_a(output, ground_truth).squeeze().mean().item()
        print("output_den's dimension:{} Ground truth's dimension:{}".format(output_den.shape, ground_truth_den.shape))
        test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | LPIPS(VGG): {lpips_score:2.4f} | LPIPS(ALEX): {lpips_score_a:2.4f} | SSIM: {ssim_score:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | ")

        ouput_dir = os.path.join(config['output_dir'], config['exp_name'], file_name)
        
        psnrs[file_name +'_psnr'] = test_psnr
        lpips_sc[file_name +'_lpips(vgg)'] = lpips_score
        lpips_sc_a[file_name +'_lpips(alex)'] = lpips_score_a
        ssim[file_name +'_ssim'] = ssim_score
        mse_i[file_name + '_mse_i'] = test_mse

        os.makedirs(os.path.join(ouput_dir), exist_ok=True)

        exp_name = config['exp_name']
        inversefed.utils.save_to_table(os.path.join(ouput_dir), name=f'{exp_name}', dryrun=args.dryrun,
                                    rec_loss=stats["opt"],
                                    psnr=test_psnr,
                                    LPIPS_VGG=lpips_score,
                                    LPIPS_ALEX=lpips_score_a,
                                    ssim=ssim_score,
                                    test_mse=test_mse,
                                    feat_mse=feat_mse,
                                    target_id=target_id,
                                    seed=model_seed
                                    )


        for j in range(config['num_images']):
            torchvision.utils.save_image(output_den[j:j + 1, ...], os.path.join(ouput_dir, f'{tid_list[j]}_gen.png'))


        # Update target id
        target_id = target_id_
            
        inversefed.utils.save_to_table(os.path.join(config['output_dir'], config['exp_name']), name='Metrics', dryrun=args.dryrun, target_id=int(target_id - 1), **psnrs, **lpips_sc, **lpips_sc_a, **ssim, **mse_i)
            

        for j in range(config['num_images']):
            torchvision.utils.save_image(ground_truth_den[j:j + 1, ...], os.path.join(config['output_dir'], config['exp_name'], f'{tid_list[j]}_gt.png'))

        if i % 5 == 0:
            torch.save(inn.state_dict(), os.path.join(os.path.join(config['output_dir'], config['exp_name']), f'INN_{i}.pth'))



        torch.save(inn.state_dict(), os.path.join(os.path.join(config['output_dir'], config['exp_name']), f'INN.pth'))


    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')