import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class SoftDiceLoss_binary(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        # print(target.shape)
        # print(target.view(-1))
        target = target.clone().view(batch_size, -1)

        inter = torch.sum(input * target, 1) + smooth
        union = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth

        score = torch.sum(2.0 * inter / union) / float(batch_size)
        score = 1.0 - torch.clamp(score, 0.0, 1.0 - 1e-7)

        return score


