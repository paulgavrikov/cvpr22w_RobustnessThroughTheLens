import torch.nn as nn

__all__ = [
    "lowres_resnet9",
    "lowres_auxresnet9",
]


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = norm_layer(num_features=out_channels)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = norm_layer(num_features=out_channels)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(num_features=out_channels)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out


class LowResResNet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, in_channels=3, num_classes=10, norm_layer=None):
        super(LowResResNet9, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )

        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out)
        return out


class LowResAuxResNet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, in_channels=3, num_classes=10, norm_layer=None):
        super(LowResAuxResNet9, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        ])
        
        self.aux_layers = nn.ModuleList([nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes)) for module in self.layers if type(module) == nn.Conv2d])

        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=False)

    def forward(self, x):
        
        aux_outs = []
        out = x
        i = 0

        for module in self.layers:
            out = module(out)
            if self.training and type(module) == nn.Conv2d:
                aux_out = self.aux_layers[i](out)
                aux_outs.append(aux_out)
                i += 1
        
        out = self.fc(out)
        
        if self.training:
            return out, aux_outs
        else:
            return out
    
def lowres_resnet9(**kwargs):
    return LowResResNet9(**kwargs)

def lowres_auxresnet9(**kwargs):
    return LowResAuxResNet9(**kwargs)