import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetModule(nn.Module):
    def __init__(self, num_classes=1, num_filters=32,
                 encoder="vgg11", pretrained=True):
        super(UNetModule, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        encoder = getattr(models, encoder)(pretrained=pretrained).features
        self.init_encoders(encoder)
        self.init_decoders()
        self.pool = nn.MaxPool2d(2,2)
        
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        elif num_classes > 1:
            self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        output = self.final(dec1)

        return self.activation(output)

    def init_encoders(self, encoders):
        convs = []
        idx = 1
        for layer in encoders:
            convs.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                setattr(self, f"conv{idx}", nn.Sequential(*convs[:-1]))
                idx += 1
                convs = []

    def init_decoders(self):
        num_filters = self.num_filters
        num_classes = self.num_classes

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5   = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4   = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3   = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2   = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1   = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final  = nn.Conv2d(num_filters, num_classes, kernel_size=1)



