import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter


class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(Vgg, self).__init__()

        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=pretrained)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=pretrained)

        pool_idx = 0

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        #        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv


def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg11_bn', ss, ks, hidden, pretrained, dropout)


def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg19_bn', ss, ks, hidden, pretrained, dropout)


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        if backbone == 'vgg11_bn':
            self.model = vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg19_bn(**kwargs)
        # elif backbone == 'resnet50':
        #     self.model = Resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
