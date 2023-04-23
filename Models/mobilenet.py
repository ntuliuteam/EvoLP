import random

import torch
import torch.nn as nn


class MobileNetV1(nn.Module):
    def __init__(self, cfg=None):
        super(MobileNetV1, self).__init__()
        self.last_cfg = cfg[-1]
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=int(inp), bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, cfg[0], 2),
            conv_dw(cfg[0], cfg[1], 1),
            conv_dw(cfg[1], cfg[2], 2),
            conv_dw(cfg[2], cfg[3], 1),
            conv_dw(cfg[3], cfg[4], 2),
            conv_dw(cfg[4], cfg[5], 1),
            conv_dw(cfg[5], cfg[6], 2),
            conv_dw(cfg[6], cfg[7], 1),
            conv_dw(cfg[7], cfg[8], 1),
            conv_dw(cfg[8], cfg[9], 1),
            conv_dw(cfg[9], cfg[10], 1),
            conv_dw(cfg[10], cfg[11], 1),
            conv_dw(cfg[11], cfg[12], 2),
            conv_dw(cfg[12], cfg[13], 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(cfg[-1], 100)

    def forward(self, x):
        x = self.model(x)
        # print('x:', x.shape)
        x = x.view(-1, self.last_cfg)
        # print(x.shape)
        x = self.fc(x)
        return x


def mobilenet_v1(cfg=None):
    if cfg is None:
        cfg = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
    model = MobileNetV1(cfg)
    return model
