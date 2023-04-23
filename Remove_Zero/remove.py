# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.
import copy
import sys
import threading

sys.path.append('../')

import Models as models
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import torchvision.models as models
# import models

import os

import numpy as np


# import numpy as np
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save', default='./save.pth.tar', type=str, help='the folder to save checkpoint')

best_acc1 = 0


def main():
    args = parser.parse_args()

    # assert 'resnet50_new' in args.arch

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if 'resnet50' in args.arch:
        skip_list = [4, 5, 8, 11, 14, 15, 18, 21, 24, 27, 28, 31, 34, 37, 40, 43, 46, 47, 50, 53]  # = id + 1
    elif 'mobilenet_v1' in args.arch:
        skip_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # = id + 1
    else:
        skip_list = []  # = id + 1

    if 'inception_v3' in args.arch:
        inceptionv3_list = ["l", "l", "l", "l", "l", "l", "5", "l", "5", "l", "l", "5", "6+8+11+12", "6+8+11+12", "l",
                            "6+8+11+12", "l", "l", "6+8+11+12", "13+15+18+19", "13+15+18+19", "l", "13+15+18+19", "l",
                            "l", "13+15+18+19", "20+22+25+26", "20+22+25+26", "l", "l", "27+30+20+22+25+26",
                            "27+30+20+22+25+26", "l", "l", "27+30+20+22+25+26", "l", "l", "l", "l", "27+30+20+22+25+26",
                            "31+34+39+40", "31+34+39+40", "l", "l", "31+34+39+40", "l", "l", "l", "l", "31+34+39+40",
                            "41+44+49+50", "41+44+49+50", "l", "l", "41+44+49+50", "l", "l", "l", "l", "41+44+49+50",
                            "51+54+59+60", "51+54+59+60", "l", "l", "51+54+59+60", "l", "l", "l", "l", "51+54+59+60",
                            "61+64+69+70", "l", "61+64+69+70", "l", "l", "l", "72+76+61+64+69+70", "72+76+61+64+69+70",
                            "l", "78", "72+76+61+64+69+70", "l", "l", "82", "72+76+61+64+69+70", "77+79+80+83+84+85",
                            "77+79+80+83+84+85", "l", "87", "77+79+80+83+84+85", "l", "l", "91", "77+79+80+83+84+85",
                            "86+88+89+92+93+94"]

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.syncbn == 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()

        else:
            model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit('=>  no resume')
    else:
        exit('=> no resume')

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    if 'inception' in args.arch:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(332),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                # normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    print('=> evaluate resumed model...')
    print('=> -------------------------------------------------------------------------------------------- <=')
    validate(val_loader, model, args)
    print('=> -------------------------------------------------------------------------------------------- <=')
    new_cfg = []
    count = 0
    cfg_def = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            count = count + 1
            if count not in skip_list:
                mask = (m.weight.data != 0).float().cuda()
                cc = int(sum(mask).cpu().numpy())
                if cc < 1:
                    mask[0] = 1
                size = m.weight.data.shape[0]
                cc = cc + (cc < 1) * 1
                new_cfg.append(cc)
                cfg_def.append(size)
                cfg_mask.append(mask.clone())
                # print(cc, )
    print(new_cfg)
    print('=> pruning ratio:\t %10.2f' % (100 * (1 - sum(new_cfg) / sum(cfg_def))), '%')
    new_model = models.__dict__[args.arch](copy.deepcopy(new_cfg))
    new_model = torch.nn.DataParallel(new_model).cuda()

    # torch.save({'cfg': new_cfg, 'state_dict': new_model.state_dict()}, './tessssfdsfasf')

    # new_model.cuda()
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    count_bn = 0
    count_conv = 0
    count_ll = 0

    # new_model.to('cpu')
    # model.to('cpu')

    for [m0, m1] in zip(model.modules(), new_model.modules()):
        # assert type(m0) == type(m1)
        if isinstance(m0, nn.BatchNorm2d):
            count_bn = count_bn + 1
            assert count_bn == count_conv
            if count_bn not in skip_list:
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if len(idx1.shape) == 0 or len(idx1) == 0:
                    idx1 = np.array([0])
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                if 'inception_v3' in args.arch:
                    if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                        end_mask = cfg_mask[layer_id_in_cfg]
                else:
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                        end_mask = cfg_mask[layer_id_in_cfg]

            else:
                if 'mobilenet_v1' in args.arch:
                    # print(m1.weight.data.shape, m0.weight.data.shape)
                    # print(start_mask)
                    idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    if len(idx1.shape) == 0 or len(idx1) == 0:
                        idx1 = np.array([0])
                    m1.weight.data = m0.weight.data[idx1].clone()
                    m1.bias.data = m0.bias.data[idx1].clone()
                    m1.running_mean = m0.running_mean[idx1].clone()
                    m1.running_var = m0.running_var[idx1].clone()
                else:
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                    m1.running_mean = m0.running_mean.clone()
                    m1.running_var = m0.running_var.clone()
                    start_mask = None

        elif isinstance(m0, nn.Conv2d):
            count_conv = count_conv + 1
            if count_conv not in skip_list:
                if start_mask is not None:
                    if 'inception_v3' in args.arch:
                        cur_start = inceptionv3_list[count_conv - 1]
                        if cur_start == 'l':
                            if count_conv - 1 == 0:
                                start_mask = torch.ones(3)
                            else:
                                start_mask = cfg_mask[count_conv - 2].clone()
                        else:
                            all_last = [int(last) for last in cur_start.split('+')]
                            start_mask = torch.cat([cfg_mask[last - 1].clone() for last in all_last]).clone()
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if len(idx1.shape) == 0 or len(idx1) == 0:
                        idx1 = np.array([0])
                    if len(idx0.shape) == 0 or len(idx0) == 0:
                        idx0 = np.array([0])
                    w = m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    # print(count_conv, m1.weight.data.shape, w.clone().shape)
                    assert m1.weight.data.shape == w.clone().shape
                    m1.weight.data = w.clone()
                else:
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    w = m0.weight.data[idx1, :, :, :].clone()
                    if len(idx1.shape) == 0 or len(idx1) == 0:
                        idx1 = np.array([0])
                    # print(m1.weight.data.shape, w.clone().shape)
                    assert m1.weight.data.shape == w.clone().shape
                    m1.weight.data = w.clone()
                if m0.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
            else:
                if 'resnet50' in args.arch and count_conv == 5:
                    start_mask = cfg_mask[0]
                if start_mask is not None:
                    if 'mobilenet_v1' in args.arch and m0.groups > 1:
                        # print(m0.groups, m1.groups)
                        # print(m1.weight.data.shape, m0.weight.data.clone().shape)
                        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                        if len(idx0.shape) == 0 or len(idx0) == 0:
                            idx0 = np.array([0])
                        w = m0.weight.data[idx0, :, :, :].clone()
                        # print(m1.weight.data.shape, w.clone().shape)
                        assert m1.weight.data.shape == w.clone().shape
                        m1.weight.data = w.clone()
                    else:
                        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                        if len(idx0.shape) == 0 or len(idx0) == 0:
                            idx0 = np.array([0])
                        w = m0.weight.data[:, idx0, :, :].clone()
                        # print(m1.weight.data.shape, w.clone().shape)
                        assert m1.weight.data.shape == w.clone().shape
                        m1.weight.data = w.clone()
                else:
                    # print(m1.weight.data.shape, m0.weight.data.clone().shape)
                    assert m1.weight.data.shape == m0.weight.data.clone().shape
                    m1.weight.data = m0.weight.data.clone()
                if m0.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0, nn.Linear):
            count_ll = count_ll + 1
            if count_ll == 1 and start_mask is not None:
                if 'vgg' in args.arch:
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    w = m0.weight.data.clone().reshape(4096, 512, 7, 7)
                    w = w[:, idx0, :, :].reshape(4096, -1)
                    assert m1.weight.data.shape == w.clone().shape
                    m1.weight.data = w.clone()
                elif 'inception_v3' in args.arch:
                    cur_start = inceptionv3_list[- 1]
                    all_last = [int(last) for last in cur_start.split('+')]
                    start_mask = torch.cat([cfg_mask[last - 1] for last in all_last]).clone()
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    # print(m0.weight.data.shape)
                    # w = m0.weight.data.clone().reshape(100, 2048, 1, 1)
                    w = m0.weight.data[:, idx0].clone()
                    # print(m1.weight.data.shape, w.clone().shape)
                    assert m1.weight.data.shape == w.clone().shape
                    m1.weight.data = w.clone()
                elif 'mobilenet_v1' in args.arch:
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    w = m0.weight.data.clone().reshape(100, 1024, 1, 1)
                    w = w[:, idx0, :, :].reshape(100, -1)
                    # print(m1.weight.data.shape, w.clone().shape)
                    assert m1.weight.data.shape == w.clone().shape
                    m1.weight.data = w.clone()
                else:
                    exit('no implement')
            else:
                # print(m1.weight.data.shape, m0.weight.data.clone().shape)
                m1.weight.data = m0.weight.data.clone()
            if m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()

    print('=> evaluate new model...')
    print('=> -------------------------------------------------------------------------------------------- <=')
    validate(val_loader, new_model, args)
    print('=> -------------------------------------------------------------------------------------------- <=')

    torch.save({'cfg': new_cfg, 'state_dict': new_model.state_dict()}, args.save)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # print(images)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, folder, save_epoch=-1, epoch=-1):
    filename = os.path.join(folder, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))
    if save_epoch > 0 and (epoch % 10 == 0 or epoch % 10 == 1):
        shutil.copyfile(filename, os.path.join(folder, 'saved_' + str(epoch) + '.pth.tar'))


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
