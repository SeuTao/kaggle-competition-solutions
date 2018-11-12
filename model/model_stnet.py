# from model_helper import *
from misc import SCSEBlock
from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F

def Stconv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)

def Stconv7x7(in_, out, bias=True):
    return nn.Conv2d(in_, out, 7, padding=3, bias=bias)

def Stconv5x5(in_, out, bias=True):
    return nn.Conv2d(in_, out, 5, padding=2, bias=bias)

def Stconv1x1(in_, out, bias=True):
    return nn.Conv2d(in_, out, 1, padding=0, bias=bias)


class StBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes):
        super(StBasicBlock, self).__init__()
        self.conv1 = Stconv3x3(inplanes, planes, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Stconv3x3(planes, planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class StSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(StSEBasicBlock, self).__init__()
        self.conv1 = Stconv3x3(inplanes, planes, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Stconv3x3(planes, planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.globalAvgPool = nn.AdaptiveAvgPool2d([1,1])
        self.fc1 = nn.Linear(in_features=planes, out_features=int(round(planes / 16)))
        self.fc2 = nn.Linear(in_features=int(round(planes / 16)), out_features=planes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)
        return out

class StDblock(nn.Module):
    def __init__(self, channel):
        super(StDblock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm2d(channel)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.bn4 = nn.BatchNorm2d(channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.bn1(self.dilate1(x)))
        dilate2_out = self.relu(self.bn2(self.dilate2(dilate1_out)))
        dilate3_out = self.relu(self.bn3(self.dilate3(dilate2_out)))
        dilate4_out = self.relu(self.bn4(self.dilate4(dilate3_out)))

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class StConvRelu(nn.Module):
    def __init__(self, in_, out, kernel_size, norm_type = None):
        super(StConvRelu,self).__init__()

        is_bias = True
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out)
            is_bias = False

        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out)
            is_bias = True

        if kernel_size == 3:
            self.conv = Stconv3x3(in_, out, is_bias)
        elif kernel_size == 7:
            self.conv = Stconv7x7(in_, out, is_bias)
        elif kernel_size == 5:
            self.conv = Stconv5x5(in_, out, is_bias)
        elif kernel_size == 1:
            self.conv = Stconv1x1(in_, out, is_bias)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.norm_type is not None:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
            x = self.activation(x)
        return x

class StDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_bottleneck = True, norm_type=None):
        super(StDecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_bottleneck:
                self.block = nn.Sequential(
                    StConvRelu(in_channels, int(in_channels/4), kernel_size=1, norm_type=norm_type),
                    StConvRelu(int(in_channels/4), middle_channels, kernel_size=3, norm_type=norm_type),
                    nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )
        else:
                self.block = nn.Sequential(
                    StConvRelu(in_channels, middle_channels, kernel_size=3, norm_type=norm_type),
                    nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.block(x)

class StNetV1(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False, is_Refine = False, is_Freeze = False, is_SCSEBlock = False, norm_type = None):
        super(StNetV1, self).__init__()

        self.num_classes = num_classes
        self.is_refine = is_Refine
        self.is_SE = is_SCSEBlock

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        if is_Freeze:
            print('Freeze!!!!!!!!!!!!!!!!!!!!')
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        if self.is_SE:
            self.center = StDecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec5_se = SCSEBlock(512 + num_filters * 8, reduction=1)
            self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec4_se = SCSEBlock(256 + num_filters * 8, reduction=1)
            self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec3_se = SCSEBlock(128 + num_filters * 8, reduction=1)
            self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, norm_type)

            self.dec2_se = SCSEBlock(64 + 64 + num_filters * 2, reduction=1)
            self.dec2 = StDecoderBlock(64 + 64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, norm_type)

            self.dec1 = StDecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, norm_type)
            self.dec0 = StConvRelu(num_filters, num_filters,3)
        else:
            self.center = StDecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
            self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
            self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)
            self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, norm_type)
            self.dec2 = StDecoderBlock(64 + 64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, norm_type)
            self.dec1 = StDecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, norm_type)
            self.dec0 = StConvRelu(num_filters, num_filters, 3)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center = self.center(self.pool(conv5))

        if self.is_SE:
            dec5_se = self.dec5_se(torch.cat([center, conv5], 1))
            dec5 = self.dec5(dec5_se)
            dec4_se = self.dec4_se(torch.cat([dec5, conv4], 1))
            dec4 = self.dec4(dec4_se)
            dec3_se = self.dec3_se(torch.cat([dec4, conv3], 1))
            dec3 = self.dec3(dec3_se)
            dec2_se = self.dec2_se(torch.cat([dec3, conv2, conv1], 1))
            dec2 = self.dec2(dec2_se)
        else:
            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2, conv1], 1))

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig

class StNetV2(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False,
                 is_deconv=False, is_Refine = False, is_Freeze = False,
                 is_SCSEBlock = False, is_bottleneck = False, norm_type = None):
        super(StNetV2, self).__init__()

        self.num_classes = num_classes
        self.is_refine = is_Refine
        self.is_SE = is_SCSEBlock

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        if is_Freeze:
            print('Freeze!!!!!!!!!!!!!!!!!!!!')
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        if self.is_SE:
            self.center = StDecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv, norm_type)

            self.dec5_se = SCSEBlock(512 + num_filters * 8, reduction=16)
            self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, is_bottleneck, norm_type)

            self.dec4_se = SCSEBlock(256 + num_filters * 8, reduction=16)
            self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv,is_bottleneck, norm_type)

            self.dec3_se = SCSEBlock(128 + num_filters * 8, reduction=16)
            self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, is_bottleneck,norm_type)

            self.dec2_se = SCSEBlock(64 + num_filters * 2, reduction=16)
            self.dec2 = StDecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, is_bottleneck,norm_type)

            self.dec1 = StDecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, is_bottleneck, norm_type)
            self.dec0 = StConvRelu(num_filters, num_filters,3)
        else:
            self.center = StDecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv, is_bottleneck, norm_type)
            self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, is_bottleneck, norm_type)
            self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv, is_bottleneck, norm_type)
            self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv, is_bottleneck, norm_type)
            self.dec2 = StDecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, is_bottleneck, norm_type)
            self.dec1 = StDecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv, is_bottleneck, norm_type)
            self.dec0 = StConvRelu(num_filters, num_filters, 3)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center = self.center(self.pool(conv5))

        if self.is_SE:
            dec5_se = self.dec5_se(torch.cat([center, conv5], 1))
            dec5 = self.dec5(dec5_se)
            dec4_se = self.dec4_se(torch.cat([dec5, conv4], 1))
            dec4 = self.dec4(dec4_se)
            dec3_se = self.dec3_se(torch.cat([dec4, conv3], 1))
            dec3 = self.dec3(dec3_se)
            dec2_se = self.dec2_se(torch.cat([dec3, conv2], 1))
            dec2 = self.dec2(dec2_se)
        else:
            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig

class StNetV2_slim(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_bottleneck = True, norm_type = 'batch_norm'):
        super(StNetV2_slim, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = StDecoderBlock(512, num_filters * 8, num_filters * 8, is_bottleneck, norm_type)
        self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8,  is_bottleneck, norm_type)
        self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8, num_filters * 8,  is_bottleneck, norm_type)
        self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4, num_filters * 2, is_bottleneck, norm_type)
        self.dec2 = StDecoderBlock(64 + num_filters * 2, num_filters * 2, num_filters * 2,  is_bottleneck, norm_type)
        self.dec1 = StDecoderBlock(num_filters * 2, num_filters, num_filters, is_bottleneck, norm_type)
        self.dec0 = StConvRelu(num_filters, num_filters, 3)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        dilate4_out = F.relu(self.dilate4(dilate3_out))

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class StNetV2_slim_ChangePool(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_bottleneck = True, norm_type = 'batch_norm'):
        super(StNetV2_slim_ChangePool, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = nn.Sequential(self.encoder.layer1,
                                   self.pool)

        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = StDecoderBlock(512, num_filters * 8, num_filters * 8, is_bottleneck, norm_type)
        self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8,  is_bottleneck, norm_type)
        self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8, num_filters * 8,  is_bottleneck, norm_type)
        self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4, num_filters * 2, is_bottleneck, norm_type)
        self.dec2 = StDecoderBlock(64 + num_filters * 2, num_filters * 2, num_filters * 2,  is_bottleneck, norm_type)
        self.dec1 = StDecoderBlock(64 + num_filters * 2, num_filters, num_filters, is_bottleneck, norm_type)

        self.hypercol_1x1 = StConvRelu(num_filters*21, num_filters, 1, norm_type='batch_norm')
        self.dec0 = StConvRelu(num_filters, num_filters, 3, norm_type='batch_norm')

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)     #1/2
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig

class StNetV2_slim_ChangePool_Dblock(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_bottleneck = True, norm_type = 'batch_norm'):
        super(StNetV2_slim_ChangePool_Dblock, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = nn.Sequential(self.encoder.layer1,
                                   self.pool)

        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.dblock = Dblock(512)

        self.center = StDecoderBlock(512, num_filters * 8, num_filters * 8, is_bottleneck, norm_type)
        self.dec5 = StDecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8,  is_bottleneck, norm_type)
        self.dec4 = StDecoderBlock(256 + num_filters * 8, num_filters * 8, num_filters * 8,  is_bottleneck, norm_type)
        self.dec3 = StDecoderBlock(128 + num_filters * 8, num_filters * 4, num_filters * 2, is_bottleneck, norm_type)
        self.dec2 = StDecoderBlock(64 + num_filters * 2, num_filters * 2, num_filters * 2,  is_bottleneck, norm_type)
        self.dec1 = StDecoderBlock(64 + num_filters * 2, num_filters, num_filters, is_bottleneck, norm_type)

        self.hypercol_1x1 = StConvRelu(num_filters*21, num_filters, 1, norm_type='batch_norm')
        self.dec0 = StConvRelu(num_filters, num_filters, 3, norm_type='batch_norm')

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)     #1/2
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        dblock = self.dblock(conv5)
        center = self.center(self.pool(dblock))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            x_out_sig = F.sigmoid(x_out)

        return x_out, x_out_sig

class Decoder(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)
            x = F.dropout2d(x, p = 0.50)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        return x

class model34_DeepSupervised(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(model34_DeepSupervised, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        # hypercol = torch.cat((
        #     d1,
        #     F.upsample(d2, scale_factor=2,mode='bilinear'),
        #     F.upsample(d3, scale_factor=4, mode='bilinear'),
        #     F.upsample(d4, scale_factor=8, mode='bilinear'),
        #     F.upsample(d5, scale_factor=16, mode='bilinear')),1)

        hypercol = F.dropout2d(d1, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')),1)

        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)

        return center_fc, x_no_empty, x_no_empty_sig, x_final, x_final_sig

class StNetV2_slim_DeepSupervised(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(StNetV2_slim_DeepSupervised, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)


        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5 = StDecoderBlock(512 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')
        self.dec1 = StDecoderBlock(32 * 2, 32, 32, True, norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(32 + 64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        dec5 = self.dec5(torch.cat([f, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))

        dec1 = self.dec1(dec2)

        hypercol = F.dropout2d(dec1, p=0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)

        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)

        return center_fc, x_no_empty, x_no_empty_sig, x_final, x_final_sig

class StNetV2_slim_DeepSupervised_Dblock(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(StNetV2_slim_DeepSupervised_Dblock, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)

        self.dblock = Dblock(512)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5 = StDecoderBlock(512 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')
        self.dec1 = StDecoderBlock(32 * 2, 32, 32, True, norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(32 + 64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)


        dblock = self.dblock(conv5)

        f = self.center(dblock)
        dec5 = self.dec5(torch.cat([f, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))

        dec1 = self.dec1(dec2)

        hypercol = F.dropout2d(dec1, p=0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)

        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)

        return center_fc, x_no_empty, x_no_empty_sig, x_final, x_final_sig

class StNetV2_slim_DeepSupervised_Dblock_SCSE(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(StNetV2_slim_DeepSupervised_Dblock_SCSE, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)

        self.dblock = Dblock(512)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5_scse = SCSEBlock(512 + 32 * 8)
        self.dec5 = StDecoderBlock(512 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec4_scse = SCSEBlock(256 + 32 * 8)
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec3_scse = SCSEBlock(128 + 32 * 8)
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')
        self.dec2_scse = SCSEBlock(64 + 32 * 2)
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')
        self.dec1 = StDecoderBlock(32 * 2, 32, 32, True, norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(32 + 64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)


        dblock = self.dblock(conv5)

        f = self.center(dblock)
        dec5 = self.dec5(self.dec5_scse(torch.cat([f, conv5], 1)))
        dec4 = self.dec4(self.dec4_scse(torch.cat([dec5, conv4], 1)))
        dec3 = self.dec3(self.dec3_scse(torch.cat([dec4, conv3], 1)))
        dec2 = self.dec2(self.dec2_scse(torch.cat([dec3, conv1], 1)))

        dec1 = self.dec1(dec2)

        hypercol = F.dropout2d(dec1, p=0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)

        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)

        return center_fc, x_no_empty, x_no_empty_sig, x_final, x_final_sig

class StNetV2_slim_Dblock(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(StNetV2_slim_Dblock, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.dblock = Dblock(512)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5 = StDecoderBlock(512 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')
        self.dec1 = StDecoderBlock(32 * 2, 32, 32, True, norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, padding=0))


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        dblock = self.dblock(conv5)

        f = self.center(dblock)
        dec5 = self.dec5(torch.cat([f, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))

        dec1 = self.dec1(dec2)

        hypercol = F.dropout2d(dec1, p=0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        return x_no_empty, x_no_empty_sig

class StNetV2_slim_DeepSupervised_slim(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(StNetV2_slim_DeepSupervised_slim, self).__init__()

        self.num_classes = num_classes
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5 = StDecoderBlock(512 + 32 * 8, 32 * 8, 32 * 8, True, True, norm_type='batch_norm')
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, True, norm_type='batch_norm')
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, True, norm_type='batch_norm')
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, True, norm_type='batch_norm')
        self.dec1 = StDecoderBlock(64, 32, 32, True, True, norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(16, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(32 + 64, 32, kernel_size=1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(16, 1, kernel_size=1))


    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        dec5 = self.dec5(torch.cat([f, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))

        dec1 = self.dec1(dec2)

        hypercol = F.dropout2d(dec1, p=0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)

        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)

        return center_fc, x_no_empty, x_no_empty_sig, x_final, x_final_sig

from senet import se_resnext50_32x4d
class model50_DeepSupervised(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(model50_DeepSupervised, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec5 = StDecoderBlock(512 + 256, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')

        self.dec1 = StDecoderBlock(64, 32, 32, True,  norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(32, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(32 + 64, 32, kernel_size=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)

        conv5 = self.dec5_1x1(conv5)
        dec5 = self.dec5(torch.cat([f, conv5], 1))

        conv4 = self.dec4_1x1(conv4)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))

        conv3 = self.dec3_1x1(conv3)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))

        conv2 = self.dec2_1x1(conv2)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        d1 = self.dec1(dec2)

        hypercol = F.dropout2d(d1, p = 0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2],mode='bilinear')),1)

        x_final = self.logits_final( hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)

        return center_fc, x_no_empty, x_no_empty_sig, x_final, x_final_sig

from senet import se_resnext50_32x4d
class model50(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(model50, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec5 = StDecoderBlock(512 + 256, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')

        self.dec1 = StDecoderBlock(64, 32, 32, True,  norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(32, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32


        f = self.center(conv5)

        conv5 = self.dec5_1x1(conv5)
        dec5 = self.dec5(torch.cat([f, conv5], 1))

        conv4 = self.dec4_1x1(conv4)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))

        conv3 = self.dec3_1x1(conv3)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))

        conv2 = self.dec2_1x1(conv2)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        d1 = self.dec1(dec2)

        hypercol = F.dropout2d(d1, p = 0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        return x_no_empty, x_no_empty_sig

from senet import se_resnext50_32x4d
class model50_Dblock_SCSE(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(model50_Dblock_SCSE, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dblock = Dblock(256)

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec5_scse = SCSEBlock(512 + 32 * 8)
        self.dec5 = StDecoderBlock(512 + 256, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec4_scse = SCSEBlock(256 + 32 * 8)
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3_scse = SCSEBlock(128 + 32 * 8)
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2_scse = SCSEBlock(64 + 32 * 2)
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')

        self.dec1 = StDecoderBlock(64, 32, 32, True,  norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(32, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        f = self.center(conv5)
        f = self.dblock(f)

        conv5 = self.dec5_1x1(conv5)
        dec5 = self.dec5(self.dec5_scse(torch.cat([f, conv5], 1)))

        conv4 = self.dec4_1x1(conv4)
        dec4 = self.dec4(self.dec4_scse(torch.cat([dec5, conv4], 1)))

        conv3 = self.dec3_1x1(conv3)
        dec3 = self.dec3(self.dec3_scse(torch.cat([dec4, conv3], 1)))

        conv2 = self.dec2_1x1(conv2)
        dec2 = self.dec2(self.dec2_scse(torch.cat([dec3, conv2], 1)))

        d1 = self.dec1(dec2)

        hypercol = F.dropout2d(d1, p = 0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        return x_no_empty, x_no_empty_sig


from senet import se_resnext101_32x4d
class model101_Dblock_SCSE(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(model101_Dblock_SCSE, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnext101_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.dblock = Dblock(256)

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec5_scse = SCSEBlock(512 + 32 * 8)
        self.dec5 = StDecoderBlock(512 + 256, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec4_scse = SCSEBlock(256 + 32 * 8)
        self.dec4 = StDecoderBlock(256 + 32 * 8, 32 * 8, 32 * 8, True, norm_type='batch_norm')

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3_scse = SCSEBlock(128 + 32 * 8)
        self.dec3 = StDecoderBlock(128 + 32 * 8, 32 * 4, 32 * 2, True, norm_type='batch_norm')

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2_scse = SCSEBlock(64 + 32 * 2)
        self.dec2 = StDecoderBlock(64 + 32 * 2, 32 * 2, 32 * 2, True, norm_type='batch_norm')

        self.dec1 = StDecoderBlock(64, 32, 32, True,  norm_type='batch_norm')

        self.logits_no_empty = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(32, 1, kernel_size=1, padding=0))



    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        f = self.center(conv5)
        f = self.dblock(f)

        conv5 = self.dec5_1x1(conv5)
        dec5 = self.dec5(self.dec5_scse(torch.cat([f, conv5], 1)))

        conv4 = self.dec4_1x1(conv4)
        dec4 = self.dec4(self.dec4_scse(torch.cat([dec5, conv4], 1)))

        conv3 = self.dec3_1x1(conv3)
        dec3 = self.dec3(self.dec3_scse(torch.cat([dec4, conv3], 1)))

        conv2 = self.dec2_1x1(conv2)
        dec2 = self.dec2(self.dec2_scse(torch.cat([dec3, conv2], 1)))

        d1 = self.dec1(dec2)

        hypercol = F.dropout2d(d1, p = 0.50)
        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        return x_no_empty, x_no_empty_sig