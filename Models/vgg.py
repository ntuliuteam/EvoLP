import torch
import torch.nn as nn


# from .utils import load_state_dict_from_url


class VGG(nn.Module):

    def __init__(self, features, num_classes=100, init_weights=True, last_cfg=512):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(last_cfg * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# cfgs = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#     # 'E':[15, 31, 74, 77, 156, 158, 150, 136, 266, 255, 263, 260, 236, 273, 329, 72]
# }


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False

    # vgg19
    if 'vgg19' in arch:
        cfg.insert(2, 'M')
        cfg.insert(5, 'M')
        cfg.insert(10, 'M')
        cfg.insert(15, 'M')
        cfg.insert(20, 'M')
    elif 'vgg16' in arch:
        cfg.insert(2, 'M')
        cfg.insert(5, 'M')
        cfg.insert(9, 'M')
        cfg.insert(13, 'M')
        cfg.insert(17, 'M')
    else:
        exit('no arch')
    model = VGG(make_layers(cfg, batch_norm=batch_norm), last_cfg=cfg[-2], **kwargs)

    if pretrained:
        print('No Pretrained!!!')
        # state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        # model.load_state_dict(state_dict)

    return model


def vgg19(cfg=None, batch_norm=True, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if cfg is None:
        cfg = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    return _vgg('vgg19_bn_2', cfg, batch_norm, pretrained, progress, **kwargs)


def vgg16(cfg=None, batch_norm=True, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration 'D') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if cfg is None:
        cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    return _vgg('vgg16_bn', cfg, batch_norm, pretrained, progress, **kwargs)
