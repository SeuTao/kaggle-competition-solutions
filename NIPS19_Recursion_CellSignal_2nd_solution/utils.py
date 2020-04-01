from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import math
import argparse
import pprint
import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / (self.num_classes*1.0)
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)

    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def metric(logit, truth, is_average=True, is_prob = False, topn =5):

    if is_prob:
        prob = logit
    else:
        prob = F.softmax(logit, 1)

    value, top = prob.topk(topn, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    if is_average==True:
        # top-3 accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct/len(truth)

        top = [correct[0],
               correct[0] + correct[1],
               correct[0] + correct[1] + correct[2],
               correct[0] + correct[1] + correct[2] + correct[3],
               correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]

        precision = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5

        return precision, top
    else:
        correct = correct.float().sum(0, keepdim=False)
        correct = correct/len(truth)
        return correct

def top_n_np(preds, labels):
    n = 5
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    top5 = []

    re = 0
    for i in range(len(preds)):
        predicted_tmp = predicted[i]
        labels_tmp = labels[i]
        for n_ in range(5):
            re += np.sum(labels_tmp == predicted_tmp[n_]) / (n_ + 1.0)

    re = re / len(preds)

    for i in range(n):
        top5.append(np.sum(labels == predicted[:, i])/ (1.0*len(labels)))

    return re, top5

def get_center(vectors):
    avg = np.mean(vectors, axis=0)
    if avg.ndim == 1:
        avg = avg / np.linalg.norm(avg)
    elif avg.ndim == 2:
        assert avg.shape[1] == 512
        avg = avg / np.linalg.norm(avg, axis=1, keepdims=True)
    else:
        assert False, avg.shape
    return avg

def average_features(features, id_list):
    averaged_features = []
    averaged_id_list = []

    unique_ids = set(id_list)
    unique_ids = list(sorted(list(unique_ids)))

    for unique_id in unique_ids:
        assert unique_id != 'new_whale'
        cur_features = [feature for feature, Id
                        in zip(features, id_list) if Id == unique_id]

        cur_features = np.stack(cur_features, axis=0)

        if len(cur_features) == 1:
            averaged_features.append(cur_features[0])
            averaged_id_list.append(unique_id)
        else:
            averaged_feature = get_center(cur_features)
            averaged_features.append(averaged_feature)
            averaged_id_list.append(unique_id)

    averaged_features = np.stack(averaged_features, axis=0)
    assert averaged_features.shape[0] == len(averaged_id_list)

    return averaged_features, averaged_id_list

def get_nearest_k(center, features, k, threshold):
    feature_with_dis = [(feature, np.dot(center, feature)) for feature in features]
    if len(feature_with_dis) > 10:
        distances = np.array([dis for _, dis in feature_with_dis])

    filtered = [feature for feature, dis in feature_with_dis if dis > 0.5]
    if len(filtered) < len(feature_with_dis):
        distances = np.array([feature for feature, dis in feature_with_dis if dis <= 0.5])
    if len(filtered) > k:
        return filtered
    feature_with_dis = [feature for feature, dis in sorted(feature_with_dis, key=lambda v: v[1], reverse=True)]
    return feature_with_dis[:k]

def get_image_center(features):
    if len(features) < 4:
        return get_center(features)

    for _ in range(2):
        center = get_center(features)
        features = get_nearest_k(center, features, int(len(features) * 3 / 4), 0.5)

        if len(features) < 4:
            break

    return get_center(features)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
