from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys
import torch.autograd as autograd
import torchvision.models as models

import simsiam.loader
sys.path.append('.')

from vae import *
from set import *
from apex import amp
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model


def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--dim', default=128, type=int, help='CNN_embed_dim')
    parser.add_argument('--kl', default=0.1, type=float, help='kl weight')
    parser.add_argument('--mode', default='normal', type=str, help='simclr')
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args


def reconst_images(batch_size=64, batch_num=1, dataloader=None, model=None):
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                _, gx, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(gx[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_GX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - gx[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                wandb.log({"_Batch_{batch}_RX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
    print('reconstruction complete!')


def test(epoch, model, testloader, args):
    # set model as testing mode
    model.eval()
    acc_gx_avg = AverageMeter()
    acc_rx_avg = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            if batch_idx >= 500:
                break
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(bs, -1)), p=2, dim=1)
            _, gx, _, _ = model(x)
            acc_gx = 1 - F.mse_loss(torch.div(gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / bs
            acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / bs

            acc_gx_avg.update(acc_gx.data.item(), bs)
            acc_rx_avg.update(acc_rx.data.item(), bs)
        if args.local_rank == 0:
            wandb.log({'acc_gx_avg': acc_gx_avg.avg, \
                       'acc_rx_avg': acc_rx_avg.avg}, commit=False)
            # plot progress
            print("\n| Validation Epoch #%d\t\tRec_gx: %.4f Rec_rx: %.4f " % (epoch, acc_gx_avg.avg, acc_rx_avg.avg))
            reconst_images(batch_size=64, batch_num=2, dataloader=testloader, model=model)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            print("Epoch {} model saved!".format(epoch + 1))


def main():
    args = parse()
    args.gpu = args.local_rank
    args.world_size = int(os.environ['WORLD_SIZE'])
    print('local rank:{}'.format(args.local_rank))
    print('world_size:{}'.format(args.world_size))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    torch.cuda.set_device(args.gpu)
    if args.local_rank == 0:
        wandb.init(config=args, name=args.save_dir.replace("./results/", ''))
        setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    print('\n[Phase 1] : Data Preparation')

    if args.dataset == 'imagenet100':
        normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        model = CVAE_imagenet_withbn(128, args.dim)

    transform_train = transforms.Compose(
        [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
        ]
    )

    if args.dataset == 'imagenet':
        print("| Preparing imagenet dataset...")
        sys.stdout.write("| ")
        root='/public/data1/datasets/imagenet2012'
        train_path = os.path.join(root, 'train')
        trainset = datasets.ImageFolder(root=train_path, transform=transform_train)

    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                              sampler=train_sampler, drop_last=True)
    # Model
    print('\n[Phase 2] : Model setup')
    if use_cuda:
        model.cuda()
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()},
    ], lr=args.lr, betas=(0.0, 0.9))

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - epoch / args.epochs)

    if args.amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)
    model = DDP(model, delay_allreduce=True)
    if args.local_rank == 0:
        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(args.epochs))
        print('| Initial Learning Rate = ' + str(args.lr))

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()

        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_kl = AverageMeter()
        if args.local_rank == 0:
            print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.cuda(), y.cuda().view(-1, )
            x, y = Variable(x), Variable(y)
            bs = x.size(0)

            _, gx, mu, logvar = model(x)
            optimizer.zero_grad()
            l_rec = F.mse_loss(x, gx)
            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l_kl /= bs * 3 * args.dim
            loss = l_rec + args.kl * l_kl

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            loss_avg.update(loss.data.item(), bs)
            loss_rec.update(l_rec.data.item(), bs)
            loss_kl.update(l_kl.data.item(), bs)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx
            if args.local_rank == 0:
                wandb.log({'loss': loss_avg.avg, \
                           'loss_rec': loss_rec.avg, \
                           'loss_kl': loss_kl.avg, \
                           'lr': optimizer.param_groups[0]['lr']}, step=n_iter)
                if (batch_idx + 1) % 30 == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write(
                        '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t Loss_rec: %.4f Loss_kl: %.4f'
                        % (epoch, args.epochs, batch_idx + 1,
                           len(trainloader),  loss_rec.avg, loss_kl.avg))
        scheduler.step()
        if epoch % 10 == 1:
            test(epoch, model, trainloader, args)
    wandb.finish()


if __name__ == '__main__':
    main()
