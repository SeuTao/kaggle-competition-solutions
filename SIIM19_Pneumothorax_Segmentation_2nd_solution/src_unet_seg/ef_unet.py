import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder


class EfficientNet_5_Encoder(nn.Module):

    def __init__(self):
        super(EfficientNet_5_Encoder, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b5')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in [2, 7, 12, 26]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_5_unet(nn.Module):
    def __init__(self):
        super(EfficientNet_5_unet, self).__init__()
        self.model_encoder = EfficientNet_5_Encoder()
        self.model_decoder = UnetDecoder(encoder_channels=(2048, 176, 64, 40, 24),
                                         decoder_channels=(256, 128, 64, 32, 16),
                                         final_channels=1,
                                         use_batchnorm=True,
                                         center=False,
                                         )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):

        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)

        return seg_feature


class EfficientNet_3_Encoder(nn.Module):

    def __init__(self):
        super(EfficientNet_3_Encoder, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in [1, 4, 7, 17]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_3_unet(nn.Module):
    def __init__(self):
        super(EfficientNet_3_unet, self).__init__()
        self.model_encoder = EfficientNet_3_Encoder()
        self.model_decoder = UnetDecoder(encoder_channels=(1536, 136, 48, 32, 24),
                                         decoder_channels=(256, 128, 64, 32, 16),
                                         final_channels=1,
                                         use_batchnorm=True,
                                         center=False,
                                         )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):

        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)

        return seg_feature
