import sys

sys.path.append('../')


import Models as models
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel


def validate(input_size, model):
    model.eval()

    with torch.no_grad():
        image = torch.rand(1, input_size[0], input_size[1], input_size[2])  # N* C*H*W
        image = image.cuda()
        model(image)

        torch.cuda.synchronize(torch.cuda.current_device())
        s = time.time()
        for i in range(100):
            model(image)
        torch.cuda.synchronize(torch.cuda.current_device())
        return (time.time() - s)*10


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser(description='Search Space of A PyTorch Model for Latency Predictor')
parser.add_argument('--arch', default='test', type=str, help='architecture to use')
parser.add_argument('--input_size', type=int, default=224, help='The size of the input')
parser.add_argument('--resume', default='./save.pth.tar', type=str, help='path to save the search space file')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

args = parser.parse_args()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        new_cfg = checkpoint['cfg']
        print(new_cfg)
        model = models.__dict__[args.arch](new_cfg)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit('=>  no resume')
else:
    exit('=> no resume')

lat = validate([3, args.input_size, args.input_size], model)
print(lat,' ms')
