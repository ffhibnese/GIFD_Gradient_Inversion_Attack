"""Mechanisms for image reconstruction from parameter gradients."""

import torch
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP

from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from copy import deepcopy
from tqdm import tqdm
import random

import inversefed.porting as porting

import math
import time
from inversefed.utils import project_onto_l1_ball
import defense
from inversefed.consts import STYLE_LEN
import nevergrad as ng
import numpy as np

imsize_dict = {
    'ImageNet': 224, 'I128':128, 'I64': 64, 'I32':32,
    'CIFAR10':32, 'CIFAR100':32, 'FFHQ':512, 'FFHQ64':64,
    'CA256': 256, 'CA128': 128, 'CA64': 64, 'CA32': 32, 
    'PERM64': 64, 'PERM32': 32, 'IMAGENET_IO' : 64, 'OOD_IMAGENET' : 64,
    'OOD_FFHQ' : 64
}

save_interval=100
construct_group_mean_at = 500
construct_gm_every = 100
DEFAULT_CONFIG = dict(signed=False,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      bn_stat=1e-1,
                      image_norm=1e-1,
                      z_norm=0,
                      group_lazy=1e-1,
                      init='randn',
                      lr_decay=True,

                      dataset='CIFAR10',

                      generative_model='',
                      gen_dataset='',
                      giml=False, 
                      gias_lr=0.1,
                      gias_iterations=0,
                      gifd=False,
                      steps=[],
                      lr_io=[],
                      start_layer=0,
                      end_layer=8,
                      #projection
                      do_project_gen_out=False,
                      do_project_noises=False,
                      do_project_latent=False,
                      max_radius_gen_out=[],
                      max_radius_noises=[],
                      max_radius_latent=[],
                      # The pre-trained StyleGAN checkpoint
                      ckpt=[],
                      #For algorithm choose:
                      gias=False,
                      ggl=False,
                      yin=False,
                      geiping=False,
                      cma_budget=0,
                      KLD=0,
                      #LR pace for training
                      lr_same_pace=False,
                      project=False,
                      defense_method=[],
                      defense_setting=[],
                      num_sample=10

                      )

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config

#+++++++++++++++++++++++++++++++++
#     Definition of class
#+++++++++++++++++++++++++++++++++
class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}      #sum输入的维度可以是tuple，代表依次对当前tensor的这些维度进行求和。
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])


class MappingProxy(nn.Module):
    def __init__(self,gaussian_ft):
        super(MappingProxy,self).__init__()
        self.mean = gaussian_ft["mean"]
        self.std = gaussian_ft["std"]
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.lrelu(self.std * x + self.mean)
        return x

class BNStatisticsHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        mean_var = [mean, var]

        self.mean_var = mean_var
        # must have no output

    def close(self):
        self.hook.remove()


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, device, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, G=None, bn_prior=((0.0, 1.0)) ):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.device = device
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.setup = dict(device=self.device, dtype=next(model.parameters()).dtype)
        self.num_samples = config['num_sample']  # For CMA-ES
        self.mean_std = mean_std
        self.num_images = num_images    

        #BN Statistics
        self.bn_layers = []
        if self.config['bn_stat'] > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn_layers.append(BNStatisticsHook(module))
        self.bn_prior = bn_prior
        
        #Group Regularizer
        self.do_group_mean = False
        self.group_mean = None

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.noises = [None for i in range(self.config['restarts'])]
        self.initial_noises = [None for i in range(self.config['restarts'])]
        self.gen_outs = [[None] for i in range(self.config['restarts'])]
        self.ys = [None for i in range(self.config['restarts'])]    #For biggan's cond_vector
        self.iDLG = True
        self.images = None
                
        # initialization
        if G:
            print("Loading G...")
            if self.config['generative_model'] == 'stylegan2':
                self.G, self.G_mapping, self.G_synthesis = G, G.G_mapping, G.G_synthesis  
                if self.num_gpus > 1:
                    self.G, self.G_mapping, self.G_synthesis = G, nn.DataParallel(self.G_mapping), nn.DataParallel(self.G_synthesis)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
                self.G_synthesis.random_noise()
            elif self.config['generative_model'] == 'stylegan2_io':
                self.G = G
                # if self.num_gpus > 1:
                #     self.G = nn.DataParallel(self.G)
              
                # self.G.to(self.device)
                self.G.requires_grad_(False)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']
                self.mpl = MappingProxy(torch.load('gaussian_fit.pt'))
                # if self.num_gpus > 1:
                #     self.mpl = nn.DataParallel(self.mpl)

            elif self.config['generative_model'] == 'BigGAN':
                self.G = G
                # if self.num_gpus > 1:
                #     self.G = nn.DataParallel(self.G)
                # self.G.to(self.device)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']
            elif self.config['generative_model'].startswith('stylegan2-ada'):
                if self.num_gpus > 1:
                    self.G, self.G_mapping, self.G_synthesis = G, nn.DataParallel(self.G_mapping), nn.DataParallel(self.G_synthesis)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
            else:
                self.G = G
                if self.num_gpus > 1:
                    self.G = nn.DataParallel(self.G)
                self.G.to(self.device)
                self.G.requires_grad_(True)
            self.G.eval() # Disable stochastic dropout and using batch stat.
        elif self.config['generative_model']:
            if self.config['generative_model'] == 'stylegan2':
                self.G, self.G_mapping, self.G_synthesis = porting.load_decoder_stylegan2(self.config, self.device, dataset=self.config['gen_dataset'])
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
                self.G_mapping.eval()
                self.G_synthesis.eval()
            elif self.config['generative_model'] in ['stylegan2_io']:

                self.G = porting.load_decoder_stylegan2_io(self.config)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']
                self.G.requires_grad_(False)
                self.mpl = MappingProxy(torch.load('gaussian_fit.pt')) 
                # if self.num_gpus > 1:
                #     self.mpl = nn.DataParallel(self.mpl)
            elif self.config['generative_model'] in ['DCGAN']:
                G = porting.load_decoder_dcgan(self.config, self.device)
                G = G.requires_grad_(True)
                self.G = G
            elif self.config['generative_model'] in ['DCGAN-untrained']:
                G = porting.load_decoder_dcgan_untrained(self.config, self.device, dataset=self.config['gen_dataset'])
                G = G.requires_grad_(True)
                self.G = G
            elif self.config['generative_model'] in ['BigGAN']:
                self.G = porting.load_decoder_biggan_io(config)
                self.G = self.G.requires_grad_(False)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']

            # print(self.G)
            self.G.eval()
        else:
            self.G = None
        
        # if torch.cuda.device_count() > 1 and self.G:
        # # G = nn.DataParallel(G)
        #     self.G = nn.DataParallel(self.G)

        if self.config['gifd']:
            self.G_io = deepcopy(self.G)
            # if torch.cuda.device_count() > 1:
            #     self.G_io = nn.DataParallel(self.G_io)
            self.project = self.config["project"]
            self.steps = self.config["steps"]
            self.gifd_loss = self.config['cost_fn']
            
        if self.config['gias']:
            self.G_list2d = [None for _ in range(self.config['restarts'])]
            for trial in range(self.config['restarts']):
                self.G_list2d[trial] = [deepcopy(self.G) for _ in range(self.num_images)]

        self.max_iterations = self.config['max_iterations']
        self.cma_iterations = self.config['cma_budget']
        
        self.gias_iterations = self.max_iterations if self.config['gias_iterations'] == 0 else self.config['gias_iterations']

        self.generative_model_name = self.config['generative_model']
        self.initial_z = None

    def set_initial_z(self, z):
        self.initial_z = z

    def init_dummy_z(self, G, generative_model_name, num_images):
        if self.initial_z is not None:
            dummy_z = self.initial_z.clone().unsqueeze(0) \
                .expand(num_images, self.initial_z.shape[0], self.initial_z.shape[1]) \
                .to(self.device).requires_grad_(True)
        elif generative_model_name.startswith('stylegan2-ada'):
            dummy_z = torch.randn(num_images, 512).to(self.device)
            dummy_z = G.mapping(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
            dummy_z = dummy_z.detach().requires_grad_(True)
        elif generative_model_name == 'stylegan2':
            dummy_z = torch.randn(num_images, 512).to(self.device)
            if self.config['gen_dataset'].startswith('I'):
                num_latent_layers = 16
            else:
                num_latent_layers = 18
            dummy_z = self.G_mapping(dummy_z).unsqueeze(1).expand(num_images, num_latent_layers, 512).detach().clone().to(self.device).requires_grad_(True)
            #The author took traing noise into consideration.
            # dummy_noise = G.static_noise(trainable=True)
        #Here we don't map z into w.
        elif generative_model_name == "stylegan2_io":
            dummy_z = torch.randn(
                        (num_images, 18, 512),   # 图片张数 x 层数 x 维度    这里之所以有18个隐向量，是因为它允许18层的w出现偏离。
                        dtype=torch.float,
                        requires_grad=True, device='cuda')
            dummy_z = self.mpl(dummy_z).detach().clone().requires_grad_(True)
            # print(dummy_z.requires_grad)
        elif generative_model_name == 'BigGAN':
            dummy_z = torch.randn(num_images, 128).to(self.device).requires_grad_(True)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_z = torch.randn(num_images, 100, 1, 1).to(self.device).requires_grad_(True)
        return dummy_z

    def gen_dummy_data(self, G, generative_model_name, dummy_z, gen_outs=[None], noise=None, ys=None, img_size=-1, start_layer = 0):
        if not torch.is_tensor(dummy_z):  # CMA-ES
            # dummy_z = [torch.Tensor(z).to(self.device) for z in dummy_z]
            dummy_z = torch.Tensor(dummy_z).to(self.device)
        running_device = dummy_z.device
        if generative_model_name.startswith('stylegan2-ada'):
            # @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
            dummy_data = G(dummy_z, noise_mode='random')
        elif generative_model_name.startswith('stylegan2'):
            if generative_model_name.endswith('io'):  #gen_outs=None when searching the latent space.
                if self.config['optim'] == 'CMA-ES':
                    with torch.no_grad():
                        dummy_data, _ = G([dummy_z.float()], input_is_latent=True, noise=noise, layer_in=gen_outs[-1])
                else:
                    dummy_data, _ = G([dummy_z.float()], input_is_latent=True, noise=noise, layer_in=gen_outs[-1])
            else:
                dummy_data = G(dummy_z)
            if self.config['gen_dataset'].startswith('I'):
                kernel_size = 512 // self.image_size
            else:
                kernel_size = 1024 // self.image_size
            dummy_data = torch.nn.functional.avg_pool2d(dummy_data, kernel_size)
        elif generative_model_name in ['BigGAN']:
            if self.config['optim'] == 'CMA-ES':
                with torch.no_grad(): 
                    dummy_z = dummy_z.tanh()
                    dummy_data, _ = G(dummy_z, ys.float(), 1) if G.start_layer == 0 else G(gen_outs[-1], ys.float(), 1)
            else:
                # print("start layer:{} end_layer:{} ys:{} dummy_z's size:{}".format(start_layer, ys.shape, dummy_z.shape))
                dummy_data, _ = G(dummy_z, ys.float(), 1) if start_layer == 0 else G(gen_outs[-1], ys.float(), 1)
                # print("dummy_data:{}".format(dummy_data.shape))
                # exit()
            dummy_data = torch.nn.functional.interpolate(dummy_data, size=(self.image_size, self.image_size), mode='area')

        elif generative_model_name in ['stylegan2-ada-z']:
            dummy_data = G(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_data = G(dummy_z)
        
        dm, ds = self.mean_std
        dm, ds = dm.to(running_device), ds.to(running_device)

        dummy_data = (dummy_data + 1) / 2
        dummy_data = (dummy_data - dm) / ds
        
        return dummy_data

    def count_trainable_params(self, G=None, z=None , x=None, noise=None):
        n_z, n_G, n_x, n_noise = 0,0,0,0
        if G:
            n_z = torch.numel(z) / self.num_images if z.requires_grad else 0    # single image parameter
            print(f"z: {n_z}")
            if noise:
                for item in noise:
                    n_noise += torch.numel(item) / self.num_images if item.requires_grad else 0 
                print(f"noise:{n_noise}")
            n_G += sum(layer.numel() for layer in G.parameters() if layer.requires_grad)
            print(f"G: {n_G}")
        else:
            n_x = torch.numel(x) /self.num_images if x.requires_grad else 0
            print(f"x: {n_x}")
        self.n_trainable = n_z + n_G + n_x + n_noise

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, tol=None):
        """Reconstruct image from gradient."""
        # if labels is None:
        if torch.is_tensor(input_data[0]):  
            labels_tmp = self.infer_label(input_data, num_inputs=self.num_images)
            self.input_data = [input_data]
        else:   # mutiple gradients
            labels_tmp = [self.infer_label(grad, num_inputs=self.num_images // len(input_data)) for grad in input_data]  
            labels_tmp = torch.stack(labels_tmp).squeeze()
            self.input_data = input_data
        self.reconstruct_label = False
        print("Infer labels:{}".format(labels_tmp))
        infer_labels = [-1 for i in range(len(labels_tmp))]
        # adjust the order of labels
        for idx, label in enumerate(labels_tmp):
            if label in labels:
                infer_labels[torch.nonzero(labels == label).squeeze()] = labels_tmp[idx].clone()
        for idx, label in enumerate(labels_tmp):
            if label not in labels:
                infer_labels[infer_labels.index(-1)] = labels_tmp[idx].clone()
        infer_labels = torch.stack(infer_labels)            
        print("Infer labels in correct order:{}".format(infer_labels))
            
        self.image_size = img_shape[1]
        start_time = time.time()
        ans = []
        if self.generative_model_name:  # GAN applying
            self.init_var(infer_labels)
            old_TV = self.config['total_variation']
            dummy_z = [None for _ in range(self.config['restarts'])]
            for trial in range(self.config['restarts']):
                dummy_z[trial] = self.init_dummy_z(self.G, self.generative_model_name, self.num_images)
            self.images = self._init_images(img_shape)
            
            if dryrun:
                return None
            #GGL
            if self.config['ggl']:
                self.config['cost_fn'] = 'l2'
                self.config['optim'] = 'CMA-ES' 
                self.config['total_variation'] = -1
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1
                # self.image_project = False
                dummy_z_ggl = [z.detach().clone().to(self.device).requires_grad_(True) for z in dummy_z]
                _x = self.reconstruct_by_latentCode(dummy_z_ggl, infer_labels, img_shape, dryrun, self.cma_iterations)
                _, best_score, x_best, _ = self.choose_optimal(_x, infer_labels, dummy_z=dummy_z_ggl, dryrun=dryrun)
                stats_ggl = {}
                stats_ggl['opt'] = best_score
                ans.append(['ggl'] + [x_best, stats_ggl])

                self.config['total_variation'] = old_TV

            #GIAS
            if self.config['gias']:
                self.config['cost_fn'] = 'sim_cmpr0'
                self.config['optim'] = 'adam'
                self.config['KLD'] = -1
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1
                #latent space search
                dummy_z_gias = [z.detach().clone().to(self.device).requires_grad_(True) for z in dummy_z]
                _x = self.reconstruct_by_latentCode(dummy_z_gias, infer_labels, img_shape, dryrun, self.max_iterations)
                optimal_z, _, _, optimal_val = self.choose_optimal(_x, infer_labels, dummy_z=dummy_z_gias, dryrun=dryrun)
                # print("optimal z's shape:{} _x shape:{}".format(optimal_z.shape, _x[0].shape))
                #parameter space search
                if self.generative_model_name in ['stylegan2_io']:
                    ans.append(['gias'] + list(self.gias_param_search(optimal_z, _x, infer_labels, optimal_noise=optimal_val)))
                else:
                    ans.append(['gias'] + list(self.gias_param_search(optimal_z, _x, infer_labels, optimal_ys=optimal_val)))
                
            #GIFD
            if self.config['gifd']:
                self.config['cost_fn'] = self.gifd_loss 
                self.config['optim'] = 'adam'
                self.config['KLD'] = -1
                dummy_z_io = [z.detach().clone().to(self.device).requires_grad_(True) for z in dummy_z]
                ans += self.inter_optimizer(dummy_z_io, infer_labels, -1)

 
        else:  #GAN-free method
            self.images = self._init_images(img_shape)
            if self.config['yin']:
                self.config['cost_fn'] = 'l2'
                self.config['optim'] = 'adam'
                # self.max_iterations = 1
                _x = self.reconstruct_by_latentCode(None, infer_labels, img_shape, dryrun, self.max_iterations)
                _, best_score, x_best, _ = self.choose_optimal(_x, infer_labels, dryrun=dryrun)
                stats_yin = {}
                stats_yin['opt'] = best_score
                ans.append(['Yin'] + [x_best, stats_yin])
            
            if self.config['geiping']:
                self.config['cost_fn'] = 'sim_cmpr0'
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1
                _x = self.reconstruct_by_latentCode(None, infer_labels, img_shape, dryrun, self.max_iterations)
                _, best_score, x_best, _ = self.choose_optimal(_x, infer_labels, dryrun=dryrun)
                stats_gp = {}
                stats_gp['opt'] = best_score
                ans.append(['geiping'] + [x_best, stats_gp])

        print(f'Total time: {time.time()-start_time}.')
        return ans

    def init_var(self, labels):
        if self.generative_model_name in ['stylegan2_io']:
            if not self.initial_noises[0]:  #Need noises
                noises_single = self.G.make_noise(self.num_images)
                print("Length of noises:%d" %(len(noises_single)))  
                noises = []
                for noise in noises_single:    
                    noises.append(noise.normal_().to(self.device))   
                self.initial_noises = [list(noises) for i in range(self.config['restarts'])]   #For stylegan2_io 
            
            self.noises = deepcopy(self.initial_noises)
        elif self.generative_model_name in ['BigGAN']:
            self.ys = [ torch.nn.functional.one_hot(labels, num_classes=1000).to(self.device) for i in range(self.config['restarts'])]

        self.gen_outs = [[None] for i in range(self.config['restarts'])]

    def invert_stylegan2(self, dummy_z, labels, start_layer, noise_list, steps, index):
        learning_rate = self.config['lr_io'][index]
        print(f"Running round {index + 1} / {len(self.config['steps'])} of GIFD.")


        _x = [None for _ in range(self.config['restarts'])]        
        for trial in range(self.config['restarts']):
        # noise_list contains the indices of nodes that we will be optimizing over
            for i in range(len(self.noises[trial])):   
                if i in noise_list:
                    self.noises[trial][i].requires_grad = True
                else:
                    self.noises[trial][i].requires_grad = False

            with torch.no_grad():
                if start_layer == 0:
                    var_list = [dummy_z[trial]] + self.noises[trial]
                    self.count_trainable_params(G=self.G_io, z=dummy_z[0])
                else:
                    self.gen_outs[trial][-1].requires_grad = True     
                    self.count_trainable_params(G=self.G_io, z=self.gen_outs[trial][-1], noise=self.noises[trial])

                    var_list = [dummy_z[trial]] + self.noises[trial] + [self.gen_outs[trial][-1]]
                    prev_gen_out = torch.ones(self.gen_outs[trial][-1].shape, device=self.gen_outs[trial][-1].device) * self.gen_outs[trial][-1]
                prev_latent = torch.ones(dummy_z[trial].shape, device=dummy_z[trial].device) * dummy_z[trial]
                prev_noises = [torch.ones(noise.shape, device=noise.device) * noise for noise in
                                self.noises[trial]]

                # set network that we will be optimizing over
                self.G_io.start_layer = start_layer          #start_layer is: 0 1 2 3...
                self.G_io.end_layer = self.config['end_layer']
            
            print(f"Total number of trainable parameters: {self.n_trainable}")

            optimizer = torch.optim.Adam(var_list, lr=learning_rate)
            ps = SphericalOptimizer([dummy_z[trial]] + self.noises[trial])  #pgd
            pbar = tqdm(range(steps))

            for i in pbar:
                if self.config['lr_same_pace']:
                    total_steps = sum(self.steps)
                    t = i / total_steps
                else:
                    t = i / steps
                lr = self.get_lr(t, learning_rate)
                optimizer.param_groups[0]["lr"] = lr
                _x[trial] = self.gen_dummy_data(self.G_io, self.config['generative_model'], dummy_z[trial], gen_outs=self.gen_outs[trial], noise=self.noises[trial]) 


                #-                      Calculate loss                           -#
                losses = [0, 0, 0, 0, 0]  
                optimizer.zero_grad()
                
                closure = self._gradient_closure(optimizer, _x[trial], self.input_data, labels, losses)
                rec_loss = closure()

                optimizer.step()

                if self.project:
                    ps.step()      

        #project back
                if start_layer != 0 and self.config['do_project_gen_out']:
                    if self.config['max_radius_gen_out'][index] > 0:
                        deviation = project_onto_l1_ball(self.gen_outs[trial][-1] - prev_gen_out,
                                                            self.config['max_radius_gen_out'][index])
                        var_list[-1].data = (prev_gen_out + deviation).data
                if self.config['do_project_latent']:
                    if self.config['max_radius_latent'][index] > 0:
                        deviation = project_onto_l1_ball(dummy_z[trial] - prev_latent,
                                                            self.config['max_radius_latent'][index])
                        var_list[0].data = (prev_latent + deviation).data
                if self.config['do_project_noises']:
                    if self.config['max_radius_noises'][index] > 0:
                        deviations = [project_onto_l1_ball(noise - prev_noise,
                                                            self.config['max_radius_noises'][index]) for noise,
                                        prev_noise in zip(self.noises[trial], prev_noises)]
                        for i, deviation in enumerate(deviations):
                            var_list[i+1].data = (prev_noises[i] + deviation).data

                pbar.set_description(
                    (
                        f" Rec. loss: {rec_loss.item():7.4f} | tv: {losses[0]:7.4f} | KLD: {losses[4]:7.4f} | ImageNorm: {losses[2]:7.4f}"
                    )
                )

            # TODO: check what happens when we are in the last layer
            with torch.no_grad():
                # latent_w = self.mpl(dummy_z[trial])
                self.G_io.end_layer = self.G_io.start_layer
                intermediate_out, _  = self.G_io([dummy_z[trial]],
                                                    input_is_latent=True,
                                                    noise=self.noises[trial],
                                                    layer_in=self.gen_outs[trial][-1],
                                                    skip=None)
                self.gen_outs[trial].append(intermediate_out)   

                self.G_io.end_layer = self.config['end_layer']
            
            #project back to image
            # if self.image_project:
            dm, ds = self.mean_std  
            with torch.no_grad():
                # Project into image space
                _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

        return _x
        
    def invert_biggan(self, dummy_z, labels, start_layer, steps, index, img_size=-1):
        
        print("The start_layer:{}".format(start_layer))
        print(f"Running round {index + 1} / {len(self.config['steps'])} of GIFD.")


        learning_rate = self.config['lr_io'][index]

        _x = [None for _ in range(self.config['restarts'])] 
        # if torch.cuda.device_count() > 1:
        #     self.G_io = deepcopy(self.G)
        #     self.G_io.start_layer = start_layer
        #     self.G_io = nn.DataParallel(self.G_io)
        #     self.G_io.to(self.device)
        #         #start_layer is: 0 1 2 3...
        # else:
        self.G_io.start_layer = start_layer

        for trial in range(self.config['restarts']):
            # if torch.cuda.device_count() > 1:
            #     self.G_io = deepcopy(self.G)
            #     self.G_io.start_layer = start_layer
            #     self.G_io = nn.DataParallel(self.G_io)
            #     self.G_io.to(self.device)
            #     #start_layer is: 0 1 2 3...
            # else:
            self.G_io.start_layer = start_layer

            if start_layer == 0:
                optim_param = [dummy_z[trial]]
                ps = SphericalOptimizer([dummy_z[trial]])  #pgd
                self.count_trainable_params(G=self.G_io, z=dummy_z[0])
            else:
                self.gen_outs[trial][-1].requires_grad = True     
                self.count_trainable_params(G=self.G_io, z=self.gen_outs[trial][-1])
                optim_param =  [self.gen_outs[trial][-1]]
                prev_gen_out = torch.ones(self.gen_outs[trial][-1].shape, device=self.gen_outs[trial][-1].device) * self.gen_outs[trial][-1]
            
            print(f"Total number of trainable parameters: {self.n_trainable}")

            optimizer = torch.optim.Adam(optim_param, lr=learning_rate)

            # print("_invert z:{}".format(z.shape))
            pbar = tqdm(range(steps))
            # self.match_min = np.inf
            
            # c = torch.nn.functional.one_hot(self.labels, num_classes = self.fl_setting['num_classes']).to(self.input_gradient[0].device)


            for current_step in pbar:
                # img_gen = self.generator(z, c.float(), 1)
                lr = self.get_lr(current_step / steps, learning_rate)
                # optimizer = torch.optim.Adam([optim_param[0][select_idx]], lr=learning_rate)
                optimizer.param_groups[0]['lr'] = lr

                _x[trial] = self.gen_dummy_data(self.G_io, self.config['generative_model'], dummy_z[trial], gen_outs=self.gen_outs[trial], ys=self.ys[trial], img_size=img_size, start_layer=start_layer) 
                losses = [0, 0, 0, 0, 0]  
                optimizer.zero_grad()
                self.dummy_z = dummy_z[trial]
                closure = self._gradient_closure(optimizer, _x[trial], self.input_data, labels, losses)
                rec_loss = closure()

                optimizer.step()

                if self.project and start_layer == 0:
                    ps.step()   

                if start_layer != 0 and self.config['do_project_gen_out']:
                    if self.config['max_radius_gen_out'][index] > 0:
                        deviation = project_onto_l1_ball(self.gen_outs[trial][-1] - prev_gen_out,
                                                        self.config['max_radius_gen_out'][index])
                        self.gen_outs[trial][-1].data = (prev_gen_out + deviation).data

                pbar.set_description(
                    (
                        f" Rec. loss: {rec_loss.item():7.4f} | tv: {losses[0]:7.4f} | KLD: {losses[4]:7.4f} | ImageNorm: {losses[2]:7.4f}"
                    )
                )

            with torch.no_grad():
                # if torch.cuda.device_count() > 1:
                #     self.G_io = deepcopy(self.G)
                    # self.G_io.start_layer = start_layer
                self.G_io.end_layer = start_layer + 1
                    # self.G_io = nn.DataParallel(self.G_io)
                    # self.G_io.to(self.device)
                intermediate_out, new_ys = self.G_io(self.gen_outs[trial][-1], self.ys[trial].float(), 1)   if start_layer > 0 else self.G_io(dummy_z[trial], self.ys[trial].float(), 1)
                self.gen_outs[trial].append(intermediate_out)   
                self.ys[trial] = new_ys
                self.G_io.end_layer = self.config['end_layer']
                # self.G_io = nn.DataParallel(self.G_io)
            # if self.image_project:
            dm, ds = self.mean_std  
            with torch.no_grad():
                # Project into image space
                _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)
                _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

        return _x

    def inter_optimizer(self, dummy_z, labels, img_size, dryrun=False, prefix=''):
        self.model.eval()
        self.G_io.to(self.device)
        # if torch.is_tensor(input_data[0]):
        #     input_data = [input_data]

        res = []

        if self.generative_model_name == 'stylegan2_io' or self.config['start_layer'] > 0:
            self.config['KLD'] = 0

        print("-------------Start intermidiate space search---------------")
        best_layer_img = None
        best_layer_score = {'opt':np.inf}
        res = [[prefix + f'layer{i}', None, {'opt':-1}] for i in range(len(self.config["steps"]))]

        for i, steps in enumerate(self.config["steps"]):
            begin_layer = i + self.config['start_layer']

            if begin_layer > self.config['end_layer']:
                raise Exception('Attemping to go after end layer')
            if self.generative_model_name == 'stylegan2_io':
                _x = self.invert_stylegan2(dummy_z, labels, begin_layer, range(5 + 2 *begin_layer), int(steps), i)
            elif self.generative_model_name == 'BigGAN':
                _x = self.invert_biggan(dummy_z, labels, begin_layer, int(steps), i, img_size)
                self.config['KLD'] = 0
            #_x is not in the real image space.
            #TO DO: compute score
            stats = {}
            optimal_z, stats['opt'], opt_img, _  = self.choose_optimal(_x, labels, dummy_z, dryrun=dryrun)
            if stats['opt'] < best_layer_score['opt']:  #save the best layer output
                # best_layer_name = 'Best_' + prefix + 'output' 
                # best_layer_num = i
                best_layer_img = opt_img.detach()
                best_layer_score = dict(stats)
            res[i] = [prefix + f'layer{i}', opt_img.detach(), stats]
            res.append(['Best_' + prefix + 'first_' + str(i) + '_layer' , best_layer_img, best_layer_score])

        return res

    """
    @brief Return optimal latent code and image according to the gradient match loss
    """
    def choose_optimal(self, _x, labels, dummy_z=None, tol=None, dryrun=False, G=None):

        restarts = self.config['restarts']
        scores = torch.zeros(restarts)
        x = [None for i in range(restarts)]
        # print(f"choose_optimal label type:{type(labels)}")
        for trial in range(restarts):
            x[trial] = _x[trial].detach()
            scores[trial] = self._score_trial(x[trial], self.input_data, labels)
            if tol is not None and scores[trial] <= tol:
                break
            if dryrun:
                break
        scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')


        if G:   #For GIAS
            print('Choosing optimal G...')
            return  G[optimal_index], scores[optimal_index].item(), x[trial].clone(), None
        
        
        if self.generative_model_name in ['stylegan2_io']:
            print('Choosing optimal z and noise...')
            return dummy_z[optimal_index].detach().clone(), scores[optimal_index].item(), x[trial].clone(), self.noises[optimal_index]
        elif self.generative_model_name in ['BigGAN']:
            print('Choosing optimal z and ys...')
            return dummy_z[optimal_index].detach().clone(),  scores[optimal_index].item(), x[trial].clone(), self.ys[optimal_index]
        elif self.generative_model_name:
            print('Choosing optimal z...')
            return dummy_z[optimal_index].detach().clone(),  scores[optimal_index].item(), x[trial].clone(), None
        else:
            print('Choosing optimal x...')
            return None, scores[optimal_index].item(), x[trial].clone(), None

    def reconstruct_by_latentCode(self, dummy_z, labels, img_shape, dryrun, max_iterations=500):
        self.model.eval()


        max_iterations = max_iterations
        x = self._init_images(img_shape)
        # scores = torch.zeros(self.config['restarts'])
        
        try:
            # labels = [None for _ in range(self.config['restarts'])]
            optimizer = [None for _ in range(self.config['restarts'])]
            scheduler = [None for _ in range(self.config['restarts'])]
            _x = [None for _ in range(self.config['restarts'])]

            for trial in range(self.config['restarts']):
                _x[trial] = x[trial]

                if self.G:
                    self.G.to(self.device)
                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam([dummy_z[trial]], lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer[trial] = torch.optim.SGD([dummy_z[trial]], lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer[trial] = torch.optim.LBFGS([dummy_z[trial]])
                    elif self.config['optim'] == 'CMA-ES':
                        parametrization = ng.p.Array(init=dummy_z[trial].cpu().detach().numpy())
                        optimizer[trial] = ng.optimizers.registry['CMA'](parametrization=parametrization, budget=self.config['cma_budget'])
                    else:
                        raise ValueError()
                else:
                    _x[trial].requires_grad = True
                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam([_x[trial]], lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer[trial] = torch.optim.SGD([_x[trial]], lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer[trial] = torch.optim.LBFGS([_x[trial]])
                    else:
                        raise ValueError()

                if self.config['lr_decay'] and not self.config['optim'] == 'CMA-ES':
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                                                        milestones=[self.max_iterations // 2.667, self.max_iterations // 1.6,

                                                                                    self.max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            dm, ds = self.mean_std
            
            if self.G:
                print("Start latent space search")
                self.count_trainable_params(G=self.G, z=dummy_z[0])
            else:
                print("Start original space search")
                self.count_trainable_params(x=_x[0])
            print(f"Total number of trainable parameters: {self.n_trainable}")
            

            for iteration in range(max_iterations):
                for trial in range(self.config['restarts']):
                    losses = [0,0,0,0,0]
                    #Group Regularizer
                    if trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                        self.do_group_mean = True
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                    if self.do_group_mean and trial == 0 and (iteration + 1) % construct_gm_every == 0:
                        print("construct group mean")
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                    if self.G:
                        dummy_zs = dummy_z[trial]

                        if self.config['optim'] == 'CMA-ES':
                            ng_data = [optimizer[trial].ask() for _ in range(self.num_samples)]
                            _x_gen = [self.gen_dummy_data(self.G, self.generative_model_name, ng_data[i].value, noise=self.noises[trial], ys=self.ys[trial]) for i in range(self.num_samples)]

                        elif self.generative_model_name in ['stylegan2','stylegan2-ada','stylegan2-ada-untrained']:
                            _x[trial] = self.gen_dummy_data(self.G_synthesis, self.generative_model_name, dummy_zs)

                        elif self.generative_model_name in ['stylegan2_io']:
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_zs, noise=self.noises[trial])

                        elif self.generative_model_name in ['BigGAN']:  #For gias over BigGAN
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_zs, ys=self.ys[trial])
                        else:
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_zs)
                        self.dummy_z = dummy_z[trial]
                        
                    else:
                        self.dummy_z = None
                    
                    if self.config['optim'] == 'CMA-ES':
                        # ng_data = [optimizer[trial].ask() for _ in range(self.num_samples)]
                        loss = []
                        loss_detail = []
                        for i in range(self.num_samples):
                            self.dummy_z = torch.Tensor(ng_data[i].value).to(self.device)
                            closure = self._gradient_closure(optimizer[trial], _x_gen[i], self.input_data, labels, losses)
                            rec_loss = closure()
                            loss.append(rec_loss.item())
                            loss_detail.append(losses)   #record every ask's losses
                        losses = list(np.array(loss_detail).mean(axis=0))
                        rec_loss = sum(loss) / len(loss)
                        for z, l in zip(ng_data, loss):
                            optimizer[trial].tell(z, l)
                    elif self.G:
                        closure = self._gradient_closure(optimizer[trial], _x[trial], self.input_data, labels, losses)
                        rec_loss = optimizer[trial].step(closure)
                        rec_loss = rec_loss.item()
                    else:
                        imgs = _x[trial]
                        
                        closure = self._gradient_closure(optimizer[trial], imgs, self.input_data, labels, losses)
                        rec_loss = optimizer[trial].step(closure)
                        rec_loss = rec_loss.item()

                    if self.config['lr_decay'] and not self.config['optim'] == 'CMA-ES':
                        scheduler[trial].step()

                    with torch.no_grad():
                        # Project into image space

                        if (iteration + 1 == self.max_iterations) or iteration % save_interval == 0:
                            print(f'It: {iteration}. Rec. loss: {rec_loss:2.4f} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f} | kld: {losses[4]:7.4f}')
                            if self.config['z_norm'] > 0:
                                print(torch.norm(dummy_z[trial], 2).item())
                        if iteration + 1 == max_iterations and self.config['optim'] == 'CMA-ES':
                            recommendation = optimizer[trial].provide_recommendation()
                            dummy_z[trial] = torch.from_numpy(recommendation.value).float().to(self.device) 
                            with torch.no_grad():
                                _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_z[trial], noise=self.noises[trial], ys=self.ys[trial])  


                        _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                    if dryrun:
                        break

                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        if self.G:
            self.G.to("cpu")
        
        return _x

    def gias_param_search(self, optimal_z, _x, labels, optimal_noise=None, optimal_ys=None, tol=None, dryrun=False):
        
        stats = defaultdict(list)
        optimizer = [None for _ in range(self.config['restarts'])]
        scheduler = [None for _ in range(self.config['restarts'])]
        dm, ds = self.mean_std
        try:
            
            self.dummy_z = optimal_z.detach().clone().cpu()

            self.dummy_zs = [None for k in range(self.num_images)]
            
            # When optimal_noise is empty list(Biggan as generative model), self.noise_zs won't be used. 
            self.noise_zs = [None for k in range(self.num_images)]  
            self.ys_zs = [None for k in range(self.num_images)]

            if optimal_noise is not None:
                for k in range(self.num_images):
                    self.noise_zs[k] = [torch.unsqueeze(noise[k], 0)  for noise in optimal_noise]
                    if self.num_gpus > 1:
                        for i in range(len(self.noise_zs[k])):
                            self.noise_zs[k][i] = self.noise_zs[k][i].to(f'cuda:{k%self.num_gpus}')  
                            self.noise_zs[k][i].requires_grad_(False)
                    else:
                        for i in range(len(self.noise_zs[k])):
                            self.noise_zs[k][i] = self.noise_zs[k][i].to(self.device)
                            self.noise_zs[k][i].requires_grad_(False)
            # print("optimal_ys:{}".format(optimal_ys))
            if optimal_ys is not None:
                for k in range(self.num_images):
                    self.ys_zs[k] = torch.unsqueeze(optimal_ys[k], 0) 
                    if self.num_gpus > 1:                     
                        self.ys_zs[k] = self.ys_zs[k].to(f'cuda:{k%self.num_gpus}')
                    else:
                        self.ys_zs[k] = self.ys_zs[k].to(self.device)        
                    self.ys_zs[k].requires_grad_(False)
                    # print("ys_zs[k]:{}".format(self.ys_zs[k]))           
            # if optimal_ys:
            #     for k in range(self.num_images):
            #         self.ys_zs[k] = torch.unsqueeze(self.dummy_z[k], 0)
            # WIP: multiple GPUs                   
            for k in range(self.num_images):
                self.dummy_zs[k] = torch.unsqueeze(self.dummy_z[k], 0)


            
            # split generator into GPUS manually
            if self.num_gpus > 1:
                print(f"Spliting generators into {self.num_gpus} GPUs...")
                for trial in range(self.config['restarts']):
                    for k in range(self.num_images):
                        self.G_list2d[trial][k] = self.G_list2d[trial][k].to(f'cuda:{k%self.num_gpus}')
                        self.G_list2d[trial][k].requires_grad_(True)
                        self.dummy_zs[k] = self.dummy_zs[k].to(f'cuda:{k%self.num_gpus}')
                        self.dummy_zs[k].requires_grad_(False)

            else:
                for trial in range(self.config['restarts']):
                    for k in range(self.num_images):
                        self.G_list2d[trial][k] = self.G_list2d[trial][k].to(self.device)
                        self.G_list2d[trial][k].requires_grad_(True)
                        self.dummy_zs[k] = self.dummy_zs[k].to(self.device)
                        self.dummy_zs[k].requires_grad_(False)


            for trial in range(self.config['restarts']):
                if self.config['optim'] == 'adam':
                    optimizer[trial] = torch.optim.Adam([{'params': self.G_list2d[trial][k].parameters()} for k in range(self.num_images)], lr=self.config['gias_lr'])
                else:
                    raise ValueError()
    
                if self.config['lr_decay']:
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                        milestones=[self.gias_iterations // 2.667, self.gias_iterations // 1.6,
                                        self.gias_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

            #Unload G to CPU
            for trial in range(self.config['restarts']):
                for k in range(self.num_images):
                    self.G_list2d[trial][k].cpu()
            

            self.count_trainable_params(G=self.G_list2d[0][0], z=self.dummy_zs[0])
            print(f"Total number of trainable parameters: {self.n_trainable}")

            print("Start Parameter search")
            # count = 0
            for trial in range(self.config['restarts']):  #Trial is model-wise

                for k in range(self.num_images):
                    self.G_list2d[trial][k].to(f'cuda:{k%self.num_gpus}')
                for iteration in range(self.gias_iterations):
                    losses = [0,0,0,0,0]

                    _x_trial = [self.gen_dummy_data(self.G_list2d[trial][k], self.generative_model_name, self.dummy_zs[k], noise=self.noise_zs[k], ys=self.ys_zs[k]).to('cpu') for k in range(self.num_images)]
                    _x[trial] = torch.stack(_x_trial).squeeze(1).to(self.device)
                    closure = self._gradient_closure(optimizer[trial], _x[trial], self.input_data, labels, losses)
                    rec_loss = optimizer[trial].step(closure)
                    if self.config['lr_decay']:
                        scheduler[trial].step()
                    
                    with torch.no_grad():
                        # Project into image space
                        
                        _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                        if (iteration + 1 == self.gias_iterations) or iteration % save_interval == 0:
                            print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f}')

                    # Unload G to CPU
                    # for k in range(self.num_images):
                    #     self.G_list2d[trial][k].cpu()
                if dryrun:
                    break
                
                for k in range(self.num_images):
                    self.G_list2d[trial][k].cpu()
                
                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        #Unload G to CPU
        for trial in range(self.config['restarts']):
            for k in range(self.num_images):
                self.G_list2d[trial][k].cpu()

        self.G, stats['opt'], x_optimal, _ = self.choose_optimal(_x, labels, G=self.G_list2d)
        #the returned self.G is a list for a batch of imgs


        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.images is not None:
            return [img.detach().clone().to(self.device) for img in self.images]
        elif self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses):

        def closure():
            # print(f"label:{label}")
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            # print("num_images:{} num_gradients:{}".format(num_images, num_gradients))
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size

            total_loss = 0
            if self.config['optim'] != "CMA-ES":
                optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]
                loss = self.loss_fn(self.model(batch_input), batch_label)
                gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                gradient = [grad for grad in gradient]
                #apply defense
                if self.config['defense_method'] is not None:
                    if 'noise' in self.config['defense_method']:
                        # gradient = defense.additive_noise(gradient, std=self.config['defense_setting']['noise'])
                        pass
                    if 'clipping' in self.config['defense_method']:
                        gradient = defense.gradient_clipping(gradient, bound=self.config['defense_setting']['clipping'])
                    if 'compression' in self.config['defense_method']:
                        gradient = defense.gradient_compression(gradient, percentage=self.config['defense_setting']['compression'])
                    if 'representation' in self.config['defense_method']: # for ResNet
                        mask = input_gradient[0][-2][0]!=0
                        gradient[-2] = gradient[-2] * mask
                torch.cuda.empty_cache()
                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                weights=self.config['weights'])

                if self.config['total_variation'] > 0:
                    tv_loss = TV(x_trial)
                    rec_loss += self.config['total_variation'] * tv_loss
                    losses[0] = tv_loss.item()
                if self.config['bn_stat'] > 0:
                    bn_loss = 0
                    first_bn_multiplier = 10.
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                    for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                        bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                    rec_loss += self.config['bn_stat'] * bn_loss
                    losses[1] = bn_loss.item()
                if self.config['image_norm'] > 0:
                    norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['image_norm'] * norm_loss
                    losses[2] = norm_loss.item()
                if self.do_group_mean and self.config['group_lazy'] > 0:
                    group_loss =  torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['group_lazy'] * group_loss
                    losses[3] = group_loss.item()
                if self.config['z_norm'] > 0:
                    if self.dummy_z != None:
                        z_loss = torch.norm(self.dummy_z, 2)
                        rec_loss += self.config['z_norm'] * z_loss
                if self.config['KLD'] > 0:   
                    if self.generative_model_name == 'BigGAN': 
                        KLD = -0.5 * torch.sum(1 + torch.log(torch.std(self.dummy_z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(self.dummy_z.squeeze(), axis=-1).pow(2) - torch.std(self.dummy_z.squeeze(), unbiased=False, axis=-1).pow(2))
                        rec_loss += self.config['KLD'] * KLD
                        losses[4] = KLD.item()
                total_loss += rec_loss
            if self.config['optim'] != "CMA-ES":
                total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        # print(f"score_trial label type:{type(label)}")
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            gradient = [grad for grad in gradient]
            #apply defense
            if self.config['defense_method'] is not None:
                if 'noise' in self.config['defense_method']:
                    # gradient = defense.additive_noise(gradient, std=self.config['defense_setting']['noise'])
                    pass
                if 'clipping' in self.config['defense_method']:
                    gradient = defense.gradient_clipping(gradient, bound=self.config['defense_setting']['clipping'])
                if 'compression' in self.config['defense_method']:
                    gradient = defense.gradient_compression(gradient, percentage=self.config['defense_setting']['compression'])
                if 'representation' in self.config['defense_method']: # for ResNet
                    mask = input_gradient[0][-2][0]!=0
                    gradient[-2] = gradient[-2] * mask

            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'])
            total_loss += rec_loss
        return total_loss

    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):  
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = torch.sort(last_weight_min.detach().reshape((-1,)).requires_grad_(False))[0]     # Use sort to adjust the order of labels as the same to grouth truth 
        return labels
        


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0, 
                 G=None):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images, G=G)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses):

        def closure():
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size

            total_loss = 0
            optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]

                # loss = self.loss_fn(self.model(batch_input), batch_label)
                # gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                
                gradient = loss_steps(self.model, batch_input, batch_label, loss_fn=self.loss_fn,
                                        local_steps=self.local_steps, lr=self.local_lr,
                                        use_updates=self.use_updates,
                                        batch_size=self.batch_size)

                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                weights=self.config['weights'])

                if self.config['total_variation'] > 0:
                    tv_loss = TV(x_trial)
                    rec_loss += self.config['total_variation'] * tv_loss
                    losses[0] = tv_loss
                if self.config['bn_stat'] > 0:
                    bn_loss = 0
                    first_bn_multiplier = 10.
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                    for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                        bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                    rec_loss += self.config['bn_stat'] * bn_loss
                    losses[1] = bn_loss
                if self.config['image_norm'] > 0:
                    norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['image_norm'] * norm_loss
                    losses[2] = norm_loss
                if self.do_group_mean and self.config['group_lazy'] > 0:
                    group_loss =  torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['group_lazy'] * group_loss
                    losses[3] = group_loss
                if self.config['z_norm'] > 0:
                    if self.dummy_z != None:
                        z_loss = torch.norm(self.dummy_z, 2)
                        rec_loss += 1e-3 * z_loss
                total_loss += rec_loss
            total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        self.model.zero_grad()
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            # loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = loss_steps(self.model, batch_input, batch_label, loss_fn=self.loss_fn,
                                local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'])
            total_loss += rec_loss
        return total_loss

def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())

def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    # print("The length of gradients:{}".format(len(gradients)))
    # print("The length of gradients:{}".format(len(input_gradient)))

    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        # print("Indices:{}".format(indices))
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('compressed'):
                ratio = float(cost_fn[10:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)

                trial_flatten = trial_gradient[i].flatten()
                trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * trial_mask

                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask
                costs += ((trial_compressed - input_compressed).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('sim_cmpr'):
                ratio = float(cost_fn[8:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)
                
                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask

                trial_flatten = trial_gradient[i].flatten()
                # trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                # trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * input_mask

                
                costs -= (trial_compressed * input_compressed).sum() * weights[i]
                pnorm[0] += trial_compressed.pow(2).sum() * weights[i]
                pnorm[1] += input_compressed.pow(2).sum() * weights[i]

            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                input_gradient[i].flatten(),
                                                                0, 1e-10) * weights[i]
        if cost_fn.startswith('sim'):
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        if cost_fn == 'l2':
            costs /= len(indices)
        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
