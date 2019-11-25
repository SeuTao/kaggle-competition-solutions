#============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.metrics.ranking import roc_auc_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
from models.model_unet import *
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam
from tuils.loss_function import *
import torch.nn.functional as F
from collections import OrderedDict
import warnings
import segmentation_models_pytorch as smp
warnings.filterwarnings('ignore')
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
from PIL import ImageFile
import sklearn
import copy
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder
torch.backends.cudnn.benchmark = True
import argparse


def epochVal(num_fold, model, dataLoader, loss_seg, loss_cls, c_val, val_batch_size):

    model.eval ()
    lossVal = 0
    lossValNorm = 0
    valLoss_seg = 0
    valLoss_cls = 0

    outGT = []
    outPRED = []
    outGT_cls = torch.FloatTensor().cuda()
    outPRED_cls = torch.FloatTensor().cuda()

    for i, (input, target_seg, target_cls) in enumerate (dataLoader):

        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
        target_cls = target_cls.view(-1, 1).contiguous().cuda(async=True)
        outGT_cls = torch.cat((outGT_cls, target_cls), 0)
        varInput = torch.autograd.Variable(input)
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

 
    return valLoss, valLoss_seg, valLoss_cls, auc, max_threshold_list, max_result_f1_list, precision_list, recall_list


def train_one_model(model_name, image_size):

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss', 'best_thr_with_no_mask', 'best_dice_with_no_mask']
    hheader = "[epochID, round(optimizer.state_dict()['param_groups'][0]['lr'], 6), round(epoch_time, 0), round(trainLoss, 4), round(trainLoss_seg, 3), round(trainLoss_cls, 3), round(valLoss, 3), round(valLoss_seg, 3), round(valLoss_cls, 3), round(np.mean(auc), 3), round(np.mean(max_threshold_list), 3), round(np.mean(max_result_f1_list), 3), round(best_thr_with_no_mask, 3), round(float(best_dice_with_no_mask), 3), round(best_f1_thr_seg_to_cls, 3), round(best_f1_value_seg_to_cls, 3), round(best_thrr_without_no_mask, 3), round(best_dicer_without_no_mask, 3), round(best_thrr_with_no_mask_2, 3), round(best_dicer_with_no_mask_2, 3)]"
    print(hheader)
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(hheader)
    df_all = pd.read_csv(csv_path)
    kfold_path = path_data['k_fold_path_cls']

    for num_fold in range(5):
        print(num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:

            writer = csv.writer(f)
            writer.writerow([num_fold]) 

        f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]     

        print('train dataset:', len(c_train), '  val dataset c_val_without_no_mask:', 476, '  val dataset c_val_with_no_mask:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset c_val_without_no_mask:', 476, '  val dataset c_val_with_no_mask:', len(c_val)])  
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])  

        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = eval(model_name+'()')
        model = torch.nn.DataParallel(model).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)

        def loss_cls_com(input, target):
            loss_1 = FocalLoss()
            loss_2 = torch.nn.BCEWithLogitsLoss()

            loss = loss_1(input, target) + loss_2(input, target)

            return loss

        loss_cls = FocalLoss()
        loss_seg = torch.nn.BCEWithLogitsLoss()

        trMaxEpoch = 34
        lossMIN = 100000
        val_dice_max = 0

        for epochID in range (0, trMaxEpoch):

            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 0
            trainLoss_cls = 0
            trainLoss_seg = 0

            if epochID < 30:

                if epochID != 0:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2) 
            elif epochID > 29 and epochID < 32:
                optimizer.param_groups[0]['lr'] = 1e-5
            else:
                optimizer.param_groups[0]['lr'] = 5e-6

            for batchID, (input, target_seg, target_cls) in enumerate (train_loader):
                # print(input.shape)
                if batchID == 0:
                    ss_time = time.time()
                print(str(batchID) + '/' + str(int(len(c_train)/train_batch_size)) + '     ' + str((time.time()-ss_time)/(batchID+1)), end='\r')
                varInput = torch.autograd.Variable(input)
                varTarget_seg = torch.autograd.Variable(target_seg.contiguous().cuda(async=True))  
                varTarget_cls = torch.autograd.Variable(target_cls.contiguous().cuda(async=True))  
                varOutput_cls, varOutput_seg = model(varInput)
                varTarget_seg = varTarget_seg.float()
                lossvalue_seg = loss_seg(varOutput_seg, varTarget_seg)
                trainLoss_seg = trainLoss_seg + lossvalue_seg.item()
                lossvalue_cls = loss_cls_com(varOutput_cls, varTarget_cls)
                trainLoss_cls = trainLoss_cls + lossvalue_cls.item()
                lossvalue = lossvalue_cls + lossvalue_seg
                lossTrainNorm = lossTrainNorm + 1
                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()  
                del lossvalue_seg, lossvalue_cls, lossvalue


            trainLoss_seg = trainLoss_seg / lossTrainNorm
            trainLoss_cls = trainLoss_cls / lossTrainNorm
            trainLoss = trainLoss_seg + trainLoss_cls

            if epochID%1 == 0:

                valLoss, valLoss_seg, valLoss_cls, auc, max_threshold_list, max_result_f1_list, precision_list, recall_list = epochVal(num_fold, model, val_loader, loss_seg, loss_cls, c_val, val_batch_size)

            epoch_time = time.time() - start_time

            if epochID%1 == 0:  

                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'valLoss': valLoss}, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth')

            result = [epochID, round(optimizer.state_dict()['param_groups'][0]['lr'], 6), round(epoch_time, 0), round(trainLoss, 4), round(trainLoss_seg, 3), round(trainLoss_cls, 3), round(valLoss, 3), round(valLoss_seg, 3), round(valLoss_cls, 3), round(np.mean(auc), 3), round(np.mean(max_threshold_list), 3), round(np.mean(max_result_f1_list), 3)]
            print(result)

            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(result)  

        del model    


if __name__ == '__main__':
    import json
    f1 = open('../configs/seg_path_configs.json', encoding='utf-8')
    path_data = json.load(f1)
    csv_path = path_data['train_rle_path']

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='diy_model_se_resnext50_32x4d', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=1024, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=4, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=4, help='val_batch_size')

    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='diy_model_se_resnext50_32x4d_768', help='epoch')

    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 16
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)

    backbone = args.backbone
    print(backbone)
    snapshot_path = path_data['snapshot_path'] + args.model_save_path.replace('\n', '').replace('\r', '')
    train_one_model(backbone, Image_size)

