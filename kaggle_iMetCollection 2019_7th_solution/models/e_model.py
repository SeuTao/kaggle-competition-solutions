import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNet_B0(nn.Module):
    def __init__(self,  num_classes=1000, pretrained='imagenet'):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b0')
        self.enet._fc = nn.Linear(self.enet._fc.in_features, num_classes)

    def forward(self, x):
        return self.enet(x)

class EfficientNet_B1(nn.Module):
    def __init__(self,  num_classes=1000, pretrained='imagenet'):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b1')
        self.enet._fc = nn.Linear(self.enet._fc.in_features, num_classes)

    def forward(self, x):
        return self.enet(x)

class EfficientNet_B2(nn.Module):
    def __init__(self,  num_classes=1000, pretrained='imagenet'):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b2')
        self.enet._fc = nn.Linear(self.enet._fc.in_features, num_classes)

    def forward(self, x):
        return self.enet(x)

class EfficientNet_B3(nn.Module):
    def __init__(self,  num_classes=1000, pretrained='imagenet'):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b3')
        self.enet._fc = nn.Linear(self.enet._fc.in_features, num_classes)

    def forward(self, x):
        return self.enet(x)

# EfficientNet_B3()

