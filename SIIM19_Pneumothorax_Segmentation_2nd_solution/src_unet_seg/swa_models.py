# -*- coding: UTF-8 -*-
# @author hjh

import argparse
from pathlib import Path
from tuils.swa_utils import swa

import pandas as pd
from dataset.dataset import *
import torch
from torch import nn
import segmentation_models_pytorch as smp
import cv2
import albumentations
import json

RESIZE_SIZE = 1024
train_transform = albumentations.Compose([

    albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE),
    albumentations.OneOf([
        albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        ]),
    albumentations.OneOf([
        albumentations.Blur(blur_limit=4, p=1),
        albumentations.MotionBlur(blur_limit=4, p=1),
        albumentations.MedianBlur(blur_limit=4, p=1)
        ], p=0.5),

    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT, p=1),

    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)

])


class Unet_se50_model(nn.Module):
    def __init__(self):
        super(Unet_se50_model, self).__init__()
        self.model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):
        global_features = self.model.encoder(x)

        cls_feature = global_features[0]
        cls_feature = self.avgpool(cls_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls_feature = self.cls_head(cls_feature)

        seg_feature = self.model.decoder(global_features)
        return cls_feature, seg_feature


class Unet_se101_model(nn.Module):
    def __init__(self):
        super(Unet_se101_model, self).__init__()
        self.model = smp.Unet('se_resnext101_32x4d', encoder_weights='imagenet')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):
        global_features = self.model.encoder(x)

        cls_feature = global_features[0]
        cls_feature = self.avgpool(cls_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls_feature = self.cls_head(cls_feature)

        seg_feature = self.model.decoder(global_features)
        return cls_feature, seg_feature


def load(filename, model):
    print('load {}'.format(filename))
    if args.model_num == 'deeplab_se50':
        from semantic_segmentation.network.deepv3 import DeepSRNX50V3PlusD_m1  # r
        model = DeepSRNX50V3PlusD_m1(1, None)
    elif args.model_num == 'unet_ef3':
        from ef_unet import EfficientNet_3_unet
        model = EfficientNet_3_unet()
    elif args.model_num == 'unet_ef5':
        from ef_unet import EfficientNet_5_unet
        model = EfficientNet_5_unet()
    elif args.model_num == 'unet_se50':
        model = Unet_se50_model()
    elif args.model_num == 'unet_se101':
        model = Unet_se101_model()
    else:
        print('model is Error')

    model = torch.nn.DataParallel(model)
    pretrained_model_path = filename
    state = torch.load(pretrained_model_path)
    data = state['state_dict']
    model.load_state_dict(data)

    return model


import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

f1 = open('../configs/seg_path_configs.json', encoding='utf-8')
path_data = json.load(f1)
csv_path = path_data['train_rle_path']

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help='input directory which contains models')
    parser.add_argument("-o", "--output", type=str, default='swa_model.pth.tar', help='output model file')
    parser.add_argument("-e0", "--epoch0", type=int, default=4, help='choose epoch to swa')
    parser.add_argument("-e1", "--epoch1", type=int, default=14, help='choose epoch to swa')
    parser.add_argument("-e2", "--epoch2", type=int, default=19, help='choose epoch to swa')
    parser.add_argument("-mn", "--model_num", type=str, default='unet_se50', help='choose model')
    parser.add_argument("-bs", "--batch-size", type=int, default=4, help='batch size')
    parser.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')

    args = parser.parse_args()

    print(args.input)
    print(args.output)
    print(args.model_num)
    print(args.epoch0)
    print(args.epoch1)
    print(args.epoch2)
    print('bs is', args.batch_size)

    df_all = pd.read_csv(csv_path)
    kfold_path = path_data['k_fold_path']
    args.input = path_data['snapshot_path'] + args.input

    if args.model_num == 'deeplab_se50':
        from semantic_segmentation.network.deepv3 import DeepSRNX50V3PlusD_m1  # r
        model = DeepSRNX50V3PlusD_m1(1, None)
    elif args.model_num == 'unet_ef3':
        from ef_unet import EfficientNet_3_unet
        model = EfficientNet_3_unet()
    elif args.model_num == 'unet_ef5':
        from ef_unet import EfficientNet_5_unet
        model = EfficientNet_5_unet()
    elif args.model_num == 'unet_se50':
        model = Unet_se50_model()
    elif args.model_num == 'unet_se101':
        model = Unet_se101_model()
    else:
        model = None
        print('model is Error')

    for f_fold in range(5):
        num_fold = f_fold
        print(num_fold)

        f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
        c_train = f_train.readlines()
        f_train.close()
        c_train = [s.replace('\n', '') for s in c_train]
        train_dataset = Siim_Dataset(df_all, c_train, train_transform)
        net = swa(load, model, args.input, train_dataset, args.batch_size, args.device,
                  args.epoch0, args.epoch1, args.epoch2, num_fold)

        output_file = args.output.format(num_fold)
        print(output_file)
        torch.save({'epoch': 39, 'state_dict': net.state_dict(), 'valLoss': 0,
                    'best_thr_with_no_mask': 0, 'best_dice_with_no_mask': float(0),
                    'best_thr_without_no_mask': 0,
                    'best_dice_without_no_mask': float(0)}, output_file)  # save path

        print('This {} fold processing ending...'.format(num_fold))



