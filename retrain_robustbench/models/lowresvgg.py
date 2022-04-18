import os

import torch
import torch.nn as nn

__all__ = [
    "LowResVGG",
    "lowres_vgg11",
    "lowres_vgg13",
    "lowres_vgg16",
    "lowres_vgg19",
    "lowres_vgg11_bn",
    "lowres_vgg13_bn",
    "lowres_vgg16_bn",
    "lowres_vgg19_bn",
]


class LowResVGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True, in_channels=0):
        super(LowResVGG, self).__init__()

        del in_channels  # unused

        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(cfg, batch_norm, **kwargs):
    kwargs["init_weights"] = True
    model = LowResVGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=kwargs["in_channels"]), **kwargs)
    return model


def lowres_vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return _vgg("A", True, **kwargs)


def lowres_vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return _vgg("B", True, **kwargs)


def lowres_vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return _vgg("D", True, **kwargs)


def lowres_vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return _vgg("E", True, **kwargs)


def lowres_vgg11(**kwargs):
    """VGG 11-layer model (configuration "A") without batch normalization"""
    return _vgg("A", False, **kwargs)


def lowres_vgg13(**kwargs):
    """VGG 13-layer model (configuration "B") without batch normalization"""
    return _vgg("B", False, **kwargs)


def lowres_vgg16(**kwargs):
    """VGG 16-layer model (configuration "D") without batch normalization"""
    return _vgg("D", False, **kwargs)


def lowres_vgg19(**kwargs):
    """VGG 19-layer model (configuration 'E') without batch normalization"""
    return _vgg("E", False, **kwargs)
