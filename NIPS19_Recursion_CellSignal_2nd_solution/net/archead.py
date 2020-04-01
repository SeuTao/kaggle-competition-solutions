import types
import math
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=65, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels = None):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        onehot = torch.zeros(cos_th.size()).cuda()

        if labels is not  None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            onehot.scatter_(1, labels, 1)

        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


class CombineMarginModule(nn.Module):
    def __init__(self, in_features, out_features, s=65, m1 = 1.0, m2 = 0.1, m3 = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m2 = math.cos(m2)
        self.sin_m2 = math.sin(m2)
        self.th2 = math.cos(math.pi - m2)
        self.mm2 = math.sin(math.pi - m2) * m2

    def forward(self, inputs, labels = None):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))

        onehot = torch.zeros(cos_th.size()).cuda()
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            onehot.scatter_(1, labels, 1)

        # m2
        cos_th_m2 = cos_th * self.cos_m2 - sin_th * self.sin_m2
        cos_th_m2 = torch.where(cos_th > self.th2, cos_th_m2, cos_th - self.mm2)
        cond_v = cos_th - self.th2
        cond = cond_v <= 0
        cos_th_m2[cond] = (cos_th - self.mm2)[cond]

        # m3
        cos_th_m2_m3 = cos_th_m2 - self.m3

        outputs = onehot * cos_th_m2_m3 + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


