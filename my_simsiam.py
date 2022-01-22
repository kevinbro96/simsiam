import builtins
import math
import os
import random
import shutil
import time
import warnings
import os
import torch
import torchvision
import argparse
import sys
from torch.autograd import Variable
import numpy as np
import wandb
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models_adv as models
from vae import *
from apex import amp
import copy
import simsiam.loader
import simsiam.builder
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse():
    parser = argparse.ArgumentParser(description=' Seen Testing Category Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50_imagenet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50_imagenet)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 512), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # simsiam specific configs:
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')
    parser.add_argument('--fix-pred-lr', action='store_true',
                        help='Fix learning rate for the predictor')

    parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='batch norm momentum for advprop')
    parser.add_argument('--vae_path', default='../results/vae_dim512_kl0.1_simclr/model_epoch92.pth',
                        type=str, help='vae_path')
    parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
    parser.add_argument('--adv', default=False, action='store_true', help='idaa')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    args.gpu = args.local_rank

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True  # True
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.gpu)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.adv:
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch],
            args.dim, args.pred_dim, bn_adv_flag=True, bn_adv_momentum=args.bn_adv_momentum)
    else:
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch],
            args.dim, args.pred_dim)
    vae = CVAE_imagenet_withbn(128, 3072)
    vae.load_state_dict(torch.load(args.vae_path))
    vae.eval()
    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    model = convert_syncbn_model(model)
    model.cuda(args.gpu)
    vae.cuda(args.gpu)
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    model = DDP(model, delay_allreduce=True)
    vae =  DDP(vae, delay_allreduce=True)

    if args.local_rank == 0:
        wandb.init(config=args)
        print(model) # print model after SyncBatchNorm
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses_og = AverageMeter('Loss_orig', ':.4f')
        losses_adv = AverageMeter('Loss_adv', ':.4f')
        model.train()
        end = time.time()
        for step, ((x1, x2), _) in enumerate(train_loader):
            data_time.update(time.time() - end)
            x1 = x1.cuda(args.gpu, non_blocking=True)
            x2 = x2.cuda(args.gpu, non_blocking=True)

            # positive pair, with encoding
            if args.adv:
                x1_adv, gx = gen_adv(model, vae, x1, criterion, args)

            optimizer.zero_grad()
            p1, z1 = model(x1)
            p2, z2 = model(x2)

            loss_og = -(criterion(p1, z2.detach()).mean() + criterion(p2, z1.detach()).mean()) * 0.5
            if args.adv:
                p1_adv, z1_adv = model(x1_adv, adv=True)
                loss_adv = -(criterion(p1_adv, z2.detach()).mean() + criterion(p2, z1_adv.detach()).mean()) * 0.5
                loss = loss_og + loss_adv
            else:
                loss = loss_og
                loss_adv = loss_og

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_og.update(loss_og.item(), x1.size(0))
            losses_adv.update(loss_adv.item(), x1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0:
                if step % 10 == 0:
                    print(
                            f"[Epoch]: {epoch} [{step}/{len(train_loader)}]\t Batch_time: {batch_time.avg:.2f} Loss_og: {losses_og.avg:.3f} Loss_adv: {losses_adv.avg:.3f}")
                if step % 10 == 0:
                    wandb.log({'loss_og': losses_og.avg,
                               'loss_adv': losses_adv.avg,
                               'lr': optimizer.param_groups[0]['lr']})
        if args.local_rank == 0 and args.adv:
            reconst_images(x1, gx, x1_adv)
        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def gen_adv(model, vae, x1, criterion, args):
    x1 = x1.detach()
    p1, z1 = model(x1, adv=True)

    with torch.no_grad():
        z, gx, _, _ = vae(x1)
    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, True)
    x1_adv = adv_gx + (x1 - gx).detach()
    p1_adv, z1_adv = model(x1_adv, adv=True)
    tmp_loss = -criterion(p1_adv, z1).mean()
    tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle.data = variable_bottle.data + args.eps * sign_grad
        adv_gx = vae(variable_bottle, True)
        x1_adv = adv_gx + (x1 - gx).detach()
    x1_adv.requires_grad = False
    x1_adv.detach()
    return x1_adv, gx


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def reconst_images(x_i, gx, x_j_adv):
    grid_X = torchvision.utils.make_grid(x_i[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"X.jpg": [wandb.Image(grid_X)]}, commit=False)
    grid_GX = torchvision.utils.make_grid(gx[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"GX.jpg": [wandb.Image(grid_GX)]}, commit=False)
    grid_RX = torchvision.utils.make_grid((x_i[32:96] - gx[32:96]).data, nrow=8, padding=2, normalize=True)
    wandb.log({"RX.jpg": [wandb.Image(grid_RX)]}, commit=False)
    grid_AdvX = torchvision.utils.make_grid(x_j_adv[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"AdvX.jpg": [wandb.Image(grid_AdvX)]}, commit=False)
    grid_delta = torchvision.utils.make_grid((x_j_adv - x_i)[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"Delta.jpg": [wandb.Image(grid_delta)]}, commit=False)
    wandb.log({'l2_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).norm(dim=1)),
               'linf_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).abs().max(dim=1)[0])
               }, commit=False)


if __name__ == "__main__":
    main()
