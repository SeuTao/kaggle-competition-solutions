# ============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.metrics.ranking import roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# ============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torch.utils.data
import numpy as np
import albumentations
import torch.utils.data as data
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW
from tuils.loss_function import *
import torch.nn.functional as F
from collections import OrderedDict
import warnings
import argparse
import segmentation_models_pytorch as smp
from torch import distributed
import apex
from apex import amp

warnings.filterwarnings('ignore')
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
from PIL import ImageFile


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds + targs).sum(-1).float()
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


def calc_dice(gt_seg, pred_seg):
    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_dice_all(preds_m, ys):
    dice_all = 0
    for i in range(preds_m.shape[0]):
        pred = preds_m[i, 0, :, :]
        gt = ys[i, 0, :, :]
        #         print(np.sum(pred))
        if np.sum(gt) == 0 and np.sum(pred) == 0:
            dice_all = dice_all + 1
        elif np.sum(gt) == 0 and np.sum(pred) != 0:
            dice_all = dice_all
        else:
            dice_all = dice_all + calc_dice(gt, pred)
    return dice_all / preds_m.shape[0]


def epochVal(model, dataLoader, loss_seg, loss_cls, c_val, val_batch_size):
    model.eval()
    lossVal = 0
    lossValNorm = 0
    valLoss_seg = 0
    valLoss_cls = 0

    outGT = []
    outPRED = []
    outGT_cls = torch.FloatTensor().cuda()
    outPRED_cls = torch.FloatTensor().cuda()

    for i, (input, target_seg, target_cls) in enumerate(dataLoader):

        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val) / val_batch_size)) + '     ' + str((time.time() - ss_time) / (i + 1)),
              end='\r')
        target_cls = target_cls.view(-1, 1).contiguous().cuda(async=True)
        outGT_cls = torch.cat((outGT_cls, target_cls), 0)
        varInput = torch.autograd.Variable(input).cuda()
        varTarget_seg = torch.autograd.Variable(target_seg.contiguous().cuda(async=True))
        varTarget_cls = torch.autograd.Variable(target_cls.contiguous().cuda(async=True))
        outGT.append(varTarget_seg.data.cpu().float())
        varOutput_cls, varOutput_seg = model(varInput)
        varTarget_seg = varTarget_seg.float()
        lossvalue_seg = loss_seg(varOutput_seg, varTarget_seg)
        valLoss_seg = valLoss_seg + lossvalue_seg.item()
        lossvalue_cls = loss_cls(varOutput_cls, varTarget_cls)
        valLoss_cls = valLoss_cls + lossvalue_cls.item()
        varOutput_seg = varOutput_seg.sigmoid()
        varOutput_cls = varOutput_cls.sigmoid()

        outPRED_cls = torch.cat((outPRED_cls, varOutput_cls.data), 0)

        outPRED.append(varOutput_seg.data.cpu().float())
        lossValNorm += 1

    valLoss_seg = valLoss_seg / lossValNorm
    valLoss_cls = valLoss_cls / lossValNorm
    valLoss = valLoss_seg + valLoss_cls

    max_threshold_list, max_result_f1_list, precision_list, recall_list = search_f1(outPRED_cls, outGT_cls)

    auc = computeAUROC(outGT_cls, outPRED_cls, 1)

    predsr_with_no_mask = F.upsample(torch.cat(outPRED), size=(128, 128), mode='bilinear').numpy()
    ysr_with_no_mask = F.upsample(torch.cat(outGT), size=(128, 128), mode='bilinear').numpy()
    dicesr_with_no_mask = []
    thrs = np.arange(0.7, 0.9, 0.02)
    best_dicer_with_no_mask = 0
    best_thrr_with_no_mask = 0

    for i in thrs:
        preds_mr_with_no_mask = (predsr_with_no_mask > i)
        a = calc_dice_all(preds_mr_with_no_mask, ysr_with_no_mask)
        dicesr_with_no_mask.append(a)
    dicesr_with_no_mask = np.array(dicesr_with_no_mask)
    best_dicer_with_no_mask = dicesr_with_no_mask.max()
    best_thrr_with_no_mask = thrs[dicesr_with_no_mask.argmax()]

    index = np.sum(np.reshape(ysr_with_no_mask, (ysr_with_no_mask.shape[0], -1)), 1)
    predsr_without_no_mask = predsr_with_no_mask[index != 0, :, :, :]
    ysr_without_no_mask = ysr_with_no_mask[index != 0, :, :, :]

    dicesr_without_no_mask = []
    thrs = np.arange(0.2, 0.7, 0.02)
    best_dicer_without_no_mask = 0
    best_thrr_without_no_mask = 0

    for i in thrs:
        preds_mr_without_no_mask = (predsr_without_no_mask > i)
        a = calc_dice_all(preds_mr_without_no_mask, ysr_without_no_mask)
        dicesr_without_no_mask.append(a)
    dicesr_without_no_mask = np.array(dicesr_without_no_mask)

    best_dicer_without_no_mask = dicesr_without_no_mask.max()
    best_thrr_without_no_mask = thrs[dicesr_without_no_mask.argmax()]

    return valLoss, valLoss_seg, valLoss_cls, auc, max_threshold_list, max_result_f1_list, precision_list, recall_list, best_thrr_with_no_mask, best_dicer_with_no_mask, best_thrr_without_no_mask, best_dicer_without_no_mask

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


def train_one_model(model_name, img_size, use_chexpert, path_data):
    RESIZE_SIZE = img_size

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
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
    val_transform = albumentations.Compose([
        albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE, p=1),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss', 'best_thr_with_no_mask',
              'best_dice_with_no_mask', 'best_thr_without_no_mask', 'best_dice_without_no_mask']
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    # kfold_path = '/home/vip415/HJH/mmdetectionV2/kaggle_siim/data/cls_fold_5_all_images/'
    kfold_path = path_data['k_fold_path']
    extra_data = path_data['extra_img_csv']

    for f_fold in range(5):
        num_fold = f_fold
        print(num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:

            writer = csv.writer(f)
            writer.writerow([num_fold])

        if use_chexpert:
            df1 = pd.read_csv(csv_path)
            df2 = pd.read_csv(extra_data + 'chexpert_mask_{}.csv'.format(num_fold + 1))
            df_all = df1.append(df2, ignore_index=True)

            f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
            f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')

            f_fake = open(extra_data + '/chexpert_list_{}.txt'.format(num_fold + 1), 'r')

            c_train = f_train.readlines()
            c_val = f_val.readlines()
            c_fake = f_fake.readlines()
            c_train = c_fake + c_train

            f_train.close()
            f_val.close()
            f_fake.close()

        else:
            df_all = pd.read_csv(csv_path)
            f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
            f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')
            c_train = f_train.readlines()
            c_val = f_val.readlines()
            f_train.close()
            f_val.close()

        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]

        # c_train = c_train + c_val
        print('train dataset:', len(c_train), '  val dataset c_val_without_no_mask:', 476,
              '  val dataset c_val_with_no_mask:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset c_val_without_no_mask:', 476,
                             '  val dataset c_val_with_no_mask:', len(c_val)])
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])

        train_loader, val_loader = generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size,
                                                                   c_val, val_transform, val_batch_size, workers)
        if model_name == 'unet_se50':
            model = Unet_se50_model()
        elif model_name == 'unet_se101':
            model = Unet_se101_model()
        else:
            model = Unet_se50_model()
            print('model is error')
        model = apex.parallel.convert_syncbn_model(model).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        model = torch.nn.DataParallel(model)

        loss_cls = torch.nn.BCEWithLogitsLoss()
        loss_seg = SoftDiceLoss_binary()

        trMaxEpoch = 44
        lossMIN = 100000
        val_dice_max = 0

        for epochID in range(0, trMaxEpoch):

            start_time = time.time()
            model.train()
            trainLoss = 30
            lossTrainNorm = 0
            trainLoss_cls = 0
            trainLoss_seg = 0

            if epochID < 40:
                if epochID != 0:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2)
            elif epochID > 39 and epochID < 42:
                optimizer.param_groups[0]['lr'] = 1e-5
            else:
                optimizer.param_groups[0]['lr'] = 5e-6

            for batchID, (input, target_seg, target_cls) in enumerate(train_loader):

                if batchID == 0:
                    ss_time = time.time()
                print(str(batchID) + '/' + str(int(len(c_train) / train_batch_size)) + '     ' + str(
                    (time.time() - ss_time) / (batchID + 1)), end='\r')
                varInput = torch.autograd.Variable(input).cuda()
                varTarget_seg = torch.autograd.Variable(target_seg.contiguous().cuda(async=True))
                varTarget_cls = torch.autograd.Variable(target_cls.contiguous().cuda(async=True))

                varOutput_cls, varOutput_seg = model(varInput)
                varTarget_seg = varTarget_seg.float()

                lossvalue_seg = loss_seg(varOutput_seg, varTarget_seg)
                trainLoss_seg = trainLoss_seg + lossvalue_seg.item()
                lossvalue_cls = loss_cls(varOutput_cls, varTarget_cls)
                trainLoss_cls = trainLoss_cls + lossvalue_cls.item()
                # + lossvalue_seg  0*lossvalue_cls +
                lossvalue = lossvalue_cls + lossvalue_seg
                lossTrainNorm = lossTrainNorm + 1
                optimizer.zero_grad()
                with amp.scale_loss(lossvalue, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

            trainLoss_seg = trainLoss_seg / lossTrainNorm
            trainLoss_cls = trainLoss_cls / lossTrainNorm
            trainLoss = trainLoss_seg + trainLoss_cls

            if epochID%1 == 0:
                valLoss, valLoss_seg, valLoss_cls, auc, max_threshold_list, max_result_f1_list, precision_list, recall_list, best_thr_with_no_mask, best_dice_with_no_mask, best_thr_without_no_mask, best_dice_without_no_mask = epochVal(
                    model, val_loader, loss_seg, loss_cls, c_val, val_batch_size)
            epoch_time = time.time() - start_time

            if epochID%1 == 0:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'valLoss': valLoss,
                            'best_thr_with_no_mask': best_thr_with_no_mask,
                            'best_dice_with_no_mask': float(best_dice_with_no_mask),
                            'best_thr_without_no_mask': best_thr_without_no_mask,
                            'best_dice_without_no_mask': float(best_dice_without_no_mask)},
                           snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth.tar')

            result = [epochID, round(optimizer.state_dict()['param_groups'][0]['lr'], 6), round(epoch_time, 0),
                      round(trainLoss, 4), round(trainLoss_seg, 3), round(trainLoss_cls, 3), round(valLoss, 3),
                      round(valLoss_seg, 3), round(valLoss_cls, 3), round(np.mean(auc), 3),
                      round(np.mean(max_threshold_list), 3), round(np.mean(max_result_f1_list), 3),
                      round(best_thr_with_no_mask, 3), round(float(best_dice_with_no_mask), 3),
                      round(best_thr_without_no_mask, 3), round(float(best_dice_without_no_mask), 3)]
            print(result)
            # print(val_f1)
            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(result)

        del model


if __name__ == '__main__':
    import json
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    f1 = open('../configs/seg_path_configs.json', encoding='utf-8')
    path_data = json.load(f1)
    csv_path = path_data['train_rle_path']

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone_model", type=str, default=1024,
                        help='input directory which contains models')
    parser.add_argument("-img_size", "--Image_size", type=int, default=1024, help='input')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=4, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=2, help='val_batch_size')
    parser.add_argument("-use_chex", "--use_chexpert", type=int, default=0, help='epoch')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='unet_test', help='epoch')

    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    backbone = args.backbone_model
    workers = 4
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)
    print(backbone)
    snapshot_path = path_data['snapshot_path'] + args.model_save_path
    print(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    train_one_model(backbone, Image_size, args.use_chexpert, path_data)




