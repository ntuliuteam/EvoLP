# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.
import sys
import threading

sys.path.append('../')

import Models as models
import argparse
# import os
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


from Predictor_Gen import get_conv_info
import os
from Predictor import net_sim
import numpy as np
from Predictor_EDLAB import BpPredictor, RlLatency

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
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
parser.add_argument('--save', default='./logs', type=str, help='the folder to save checkpoint')
parser.add_argument('--syncbn', default=1, type=int, help='if need sync bn')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--ssr', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')

parser.add_argument('--zerobn', type=int, default=30, help='epoches set bn to 0')
parser.add_argument('--latency', type=float, default=35, help='latency')
parser.add_argument('--our', type=int, default=0, help='If zerobn method')
parser.add_argument('--interval', type=int, default=2, help='the interval of zero and recovery')
parser.add_argument('--device', type=str, default='tx2', help='the device')

best_acc1 = 0


def zeroBN(model, args):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            total += m.weight.data.shape[0]

    # print("zerobn total channel", total)
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)

    left_ratio = 0.0
    right_ratio = 0.96
    while left_ratio + 0.02 < right_ratio:
        prune_ratio = (left_ratio + right_ratio) / 2
        thre_index = int(total * prune_ratio)
        thre = y[thre_index]
        cur_cfg = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):

                k = m.weight.data.abs() - thre
                k = k + k.abs()
                k = k.sign()
                cc = int(sum(k != 0).cpu().numpy())
                if cc < 1:
                    cc = 1
                cur_cfg.append(cc)
        model_new = models.__dict__[args.arch](cur_cfg)
        conv_info = get_conv_info(model_new, 224)
        del model_new
        predictor_latency = args.predictor_.predict(conv_info)

        correct_pre_latency = 0
        if args.coeff_ is not None:
            for coeffi in args.coeff_:
                correct_pre_latency = correct_pre_latency * predictor_latency + coeffi

        if correct_pre_latency > 0:
            if correct_pre_latency > args.latency:
                left_ratio = prune_ratio
            elif correct_pre_latency < 0.97 * args.latency:
                right_ratio = prune_ratio
            else:
                break
        else:
            if predictor_latency > args.latency:
                left_ratio = prune_ratio
            elif predictor_latency < 0.97 * args.latency:
                right_ratio = prune_ratio
            else:
                break

    if args.appendx:
        args.x_.append(predictor_latency)

    thre_index = int(total * prune_ratio)
    thre = y[thre_index]

    cur_cfg = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            k = m.weight.data.abs() - thre
            k = k + k.abs()
            k = k.sign()
            cc = int(sum(k != 0).cpu().numpy())
            cur_cfg.append(cc)
            m.weight.data = m.weight.data * k
            m.bias.data = m.bias.data * k
    if 'get_real_latency_thread' in args:
        if not args.get_real_latency_thread.is_alive():
            print('r:', args.lat, args.x_[-1])
            args.y_.append(args.lat)
            args.coeff_ = np.polyfit(args.x_, args.y_, 1)
            del args.get_real_latency_thread
            args.appendx = True
    else:
        # print(cur_cfg)
        for cur_i in range(len(cur_cfg)):
            if cur_cfg[cur_i] == 0:
                cur_cfg[cur_i] = 1
        # print(cur_cfg)
        args.get_real_latency_thread = threading.Thread(target=args.real_.run_model, args=(cur_cfg, args))
        args.get_real_latency_thread.start()
        args.appendx = False


def main(times):
    args = parser.parse_args()
    args.save = os.path.join(args.save, str(times))

    args.mode = 1

    print(args.zerobn)
    if args.our != 0:
        args.coeff_ = None
        if args.device == 'tx2':
            if args.arch == 'vgg16':
                args.x_ = [101.3395932, 85.42818457, 70.14591048, 64.84928217, 60.19949295, 55.75770227, 50.71041239,
                           44.57803644,
                           39.57282714, 35.25244605, 31.79130684, 28.1309516, 25.98485808, 23.18778157, 19.893037,
                           15.26799792,
                           13.44144529, 12.12857975, 10.3352663, 7.597532038]
                args.y_ = [114.9693608, 112.8733397, 95.63884735, 81.2725544, 76.95995569, 78.03250551, 65.95778465,
                           60.08725166, 54.37976122,
                           46.90418243, 41.75333977, 38.25987577, 34.9609375, 32.40382671, 29.32010889, 39.07694817,
                           37.72929907, 28.12962532,
                           18.93223524, 10.13344526]
            else:
                exit('no x y')

            # The least squares fit is sufficiently accurate
            args.coeff_ = np.polyfit(args.x_, args.y_, 1)

            args.predictor_ = BpPredictor(mat_folder='../Predictor/' + str(args.arch) + '_tx2', device_name='tx2',
                                          model_name=str(args.arch))

            args.real_ = RlLatency(ip='192.168.0.2', model_name=str(args.arch))

        elif args.device == 'nano':
            if args.arch == 'vgg16':
                args.x_ = [160.4428892, 151.184155, 140.8803955, 130.0838496, 118.9624637, 107.3361942, 96.46361924,
                           86.49747262, 76.09936852,
                           66.51425346, 53.9403412, 45.98468903, 38.27053286, 33.02102362, 28.17934868, 24.61967641,
                           17.74760892, 14.61242728,
                           13.10075732, 10.98223596]
                args.y_ = [180.2496433, 178.2620192, 166.1232829, 149.7872949, 143.567884, 135.3099942, 114.8334026,
                           105.005908, 91.1945343,
                           83.70337486, 70.50403357, 61.44200563, 53.66722345, 47.76939154, 41.62085056, 34.3026638,
                           28.84153128, 23.86735678,
                           18.04735661, 14.12729025]
            else:
                exit('no x y')
            args.coeff_ = np.polyfit(args.x_, args.y_, 1)

            args.predictor_ = BpPredictor(mat_folder='../Predictor/' + str(args.arch) + '_nano', device_name='nano',
                                          model_name=str(args.arch))

            args.real_ = RlLatency(ip='192.168.0.3', model_name=str(args.arch), device='nano')

        args.appendx = True

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # args.times = times

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.

        if args.syncbn == 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)

        if args.our == 1:
            if epoch == args.zerobn or epoch == args.zerobn + 1:
                best_acc1 = 0.0

            if epoch <= args.zerobn or epoch % args.interval == args.mode or epoch == args.epochs - 1:
                print('zero epoch:\t', epoch)
                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                # if args.times == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, args.save, save_epoch=1, epoch=epoch)
        else:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            # if args.times == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.save, save_epoch=1, epoch=epoch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.sr:

            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                    m.weight.grad.data.add_(args.ssr * torch.sign(m.weight.data))  # L1

        optimizer.step()

        if args.our and epoch >= args.zerobn and (epoch % args.interval == args.mode or epoch == args.epochs - 1):
            zeroBN(model, args)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
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
        print(args.zerobn)

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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    ad = int(args.epochs / 3)
    lr = args.lr * (0.1 ** (epoch // ad))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    main(0)
