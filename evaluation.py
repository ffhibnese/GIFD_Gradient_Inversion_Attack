import os


import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.autograd import Variable

from PIL import Image

import lpips
import argparse
import glob
from math import exp

loss_fn = None

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def ssim_gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def ssim_create_window(window_size, channel):
    _1D_window = ssim_gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = ssim_create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return ssim_ssim(img1, img2, window, window_size, channel, size_average)

def ssim_batch(ref_batch, img_batch, batched=False, factor=1.0):

    [B, C, m, n] = img_batch.shape
    ssims = []
    for sample in range(B):
        ssims.append(ssim(img_batch.detach()[sample, :, :, :].unsqueeze(0), ref_batch[sample, :, :, :].unsqueeze(0)))
    
    mean_ssim = torch.stack(ssims, dim=0).mean()
    return mean_ssim.item(), ssims

def ssim_permute(ref_batch, img_batch, batched=False, factor=1.0):
    ### SSIM regarding permutation ### 
    ssims = []
    for i in range (img_batch.shape[0]):
        img_repeat = img_batch[i].unsqueeze(0).repeat(img_batch.shape[0], 1, 1, 1)
        _, candidate_ssims = ssim_batch(ref_batch, img_repeat)
        mx = torch.max(torch.stack(candidate_ssims).view(1, -1))
        ssims.append(mx)

    mean_ssim = torch.stack(ssims).mean()
    return mean_ssim.item(), ssims


def lpips_loss(img_batch, ref_batch, net='alex'):
    global loss_fn
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net=net)
    [B, C, m, n] = img_batch.shape
    lpips_losses = []
    for sample in range(B):
        lpips_losses.append(loss_fn(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
    lpips_loss = torch.stack(lpips_losses, dim=0).mean()

    return lpips_loss.item()


def setup_parser():
    parser = argparse.ArgumentParser(description='Calculate LPIPS cost from a trained model.')
    parser.add_argument('--result_path', default='', type=str, help='model result path')
    parser.add_argument('--model_hash', default='', type=str, help='model hash')
    parser.add_argument('--num_images', default=1, type=int, help='batch size')
    parser.add_argument('--comp_rate', default=.0, type=float, help='compression rate')
    parser.add_argument('--avg', action='store_true', help='XXX')
    parser.add_argument('--max', action='store_true', help='XXX')
    parser.add_argument('--min', action='store_true', help='XXX')

    return parser


def run(args):
    key = lambda x: int(x.split('/')[-1].split('.')[0])
    key_gt = lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0])
    tt = tf.ToTensor()
    recon_file_name = sorted(glob.glob(os.path.join(args.result_path, args.model_hash, "*[0-9].png")), key=key)
    gt_file_name = sorted(glob.glob(os.path.join(args.result_path, args.model_hash, "*[0-9]_gt.png")), key=key_gt)
    print(recon_file_name)
    # print(gt_files)
    recon_files = [tt(Image.open(recon_file_name[i])) for i in range(len(recon_file_name))]
    # recon_files = torch.stack(recon_files)
    gt_files = [tt(Image.open(gt_file_name[i])) for i in range(len(gt_file_name))]
    # gt_files = torch.stack(gt_files)

    recon_psnr = [psnr(recon_files[i].unsqueeze(0), gt_files[i].unsqueeze(0)) for i in range(len(recon_files))]
    recon_ssim = [ssim_permute(recon_files[i].unsqueeze(0), gt_files[i].unsqueeze(0))[1] for i in range(len(recon_files))]
    recon_lpips_loss = [lpips_loss(recon_files[i].unsqueeze(0), gt_files[i].unsqueeze(0)) for i in range(len(recon_files))]

    batch_psnr = []
    batch_ssim = []
    batch_lpips = []
    # print(recon_psnr)
    print('PNSR\tSSIM\tLPIPS')
    for j in range(len(recon_psnr)//args.num_images):
        # print(recon_file_name[j*args.num_images:(j+1)*args.num_images])
        if args.avg:
            batch_psnr.append(torch.Tensor(recon_psnr[j*args.num_images:(j+1)*args.num_images]).mean())
            batch_ssim.append(torch.Tensor(recon_ssim[j*args.num_images:(j+1)*args.num_images]).mean())
            batch_lpips.append(torch.Tensor(recon_lpips_loss[j*args.num_images:(j+1)*args.num_images]).mean())
            # print(f'{torch.Tensor(recon_psnr[j*args.num_images:(j+1)*args.num_images]).mean().item():.2f}\t{torch.Tensor(recon_ssim[j*args.num_images:(j+1)*args.num_images]).mean().item():.4f}\t{torch.Tensor(recon_lpips_loss[j*args.num_images:(j+1)*args.num_images]).mean().item():.4f}')
        elif args.max:
            batch_psnr.append(torch.Tensor(recon_psnr[j*args.num_images:(j+1)*args.num_images]).max())
            batch_ssim.append(torch.Tensor(recon_ssim[j*args.num_images:(j+1)*args.num_images]).max())
            batch_lpips.append(torch.Tensor(recon_lpips_loss[j*args.num_images:(j+1)*args.num_images]).min())
            #     print(f'{torch.Tensor(recon_psnr[j*args.num_images:(j+1)*args.num_images]).max().item():.2f}\t{torch.Tensor(recon_ssim[j*args.num_images:(j+1)*args.num_images]).max().item():.4f}\t{torch.Tensor(recon_lpips_loss[j*args.num_images:(j+1)*args.num_images]).min().item():.4f}')
        elif args.min:
            batch_psnr.append(torch.Tensor(recon_psnr[j*args.num_images:(j+1)*args.num_images]).min())
            batch_ssim.append(torch.Tensor(recon_ssim[j*args.num_images:(j+1)*args.num_images]).min())
            batch_lpips.append(torch.Tensor(recon_lpips_loss[j*args.num_images:(j+1)*args.num_images]).max())
            #     print(f'{torch.Tensor(recon_psnr[j*args.num_images:(j+1)*args.num_images]).min().item():.2f}\t{torch.Tensor(recon_ssim[j*args.num_images:(j+1)*args.num_images]).min().item():.4f}\t{torch.Tensor(recon_lpips_loss[j*args.num_images:(j+1)*args.num_images]).max().item():.4f}')
    batch_psnr = torch.stack(batch_psnr)
    batch_ssim = torch.stack(batch_ssim)
    batch_lpips = torch.stack(batch_lpips)
    # if args.avg:
    #     torch.save(batch_psnr, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'psnr', 'avg.pth']))
    #     torch.save(batch_ssim, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'ssim', 'avg.pth']))
    #     torch.save(batch_lpips, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'lpips', 'avg.pth']))
    # elif args.max:
    #     torch.save(batch_psnr, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'psnr', 'max.pth']))
    #     torch.save(batch_ssim, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'ssim', 'max.pth']))
    #     torch.save(batch_lpips, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'lpips', 'max.pth']))
    # elif args.min:
    #     torch.save(batch_psnr, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'psnr', 'min.pth']))
    #     torch.save(batch_ssim, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'ssim', 'min.pth']))
    #     torch.save(batch_lpips, "_".join([os.path.join('eval_result',args.result_path), str(args.comp_rate), str(args.num_images), 'lpips', 'min.pth']))
    print(f'{torch.Tensor(batch_psnr).mean().item():.6f}\t{torch.Tensor(batch_ssim).mean().item():.6f}\t{torch.Tensor(batch_lpips).mean().item():.6f}')


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    run(args)
