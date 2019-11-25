from tqdm import tqdm

import torch
import torch.nn.functional as F
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
import os, sys
sys.path.append(lib_path)

import torch.utils.data as data
from models.model_unet import *
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW
from tuils.loss_function import *
from collections import OrderedDict
import warnings
import sklearn
warnings.filterwarnings('ignore')
import argparse

torch.backends.cudnn.benchmark = True

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
#     model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input_dict in tqdm(loader):
        inputs, _, _ = input_dict
        inputs = inputs.cuda(async=True)
        input_var = torch.autograd.Variable(inputs)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def detach_params(model):
    for param in model.parameters():
        param.detach_()

    return model

def evaluate(loader, model):
    model.eval()

    out_pred = torch.FloatTensor().cuda()
    out_gt = torch.FloatTensor().cuda()

    for input_dict in loader:
        inputs, targets = input_dict
        inputs = inputs.cuda()
        targets = targets.cuda().float()

        logits = model(inputs)
        probabilities = torch.sigmoid(logits)

        out_pred = torch.cat((out_pred, probabilities), 0)
        out_gt = torch.cat((out_gt, targets), 0)

    eval_metric_bundle = search_f2(out_pred, out_gt)
    print('===> Best', eval_metric_bundle)
    
    return eval_metric_bundle




def epochVal(num_fold, model, dataLoader, loss_seg, loss_cls, c_val, val_batch_size, cls_weight, seg_weight):

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
        valLoss_seg = valLoss_seg + seg_weight * lossvalue_seg.item()
        lossvalue_cls = loss_cls(varOutput_cls, varTarget_cls)
        valLoss_cls = valLoss_cls + cls_weight * lossvalue_cls.item()
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

    df_all = pd.read_csv(csv_path)
    kfold_path = path_data['k_fold_path_cls']
    with open(checkpoint_path + '/swa.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['start'])  
            
    loss_cls = FocalLoss()
    loss_seg = torch.nn.BCEWithLogitsLoss()
    cls_weight, seg_weight = 0.01, 1

    for num_fold in range(5):
        print(num_fold)

        with open(checkpoint_path + '/swa.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['fold  :  0'])
        f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]
        
        c_train = c_train + c_val

        print('train dataset:', len(c_train), '  val dataset:', len(c_val))
   
        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        valLoss = 0
        val_f2 = 0
        val_f2_v2 = 0       

        checkpoints = [checkpoint_path + '/model_epoch_' + str(x) + '_' + str(num_fold) + '.pth' for x in epoch_list]
        model_1 = eval(model_name+'()')
        model_1 = torch.nn.DataParallel(model_1).cuda()
        state = torch.load(checkpoints[0])
        model_1.load_state_dict(state['state_dict'])
        epoch = state['epoch']
        best_thr_with_no_mask = state['best_thr_with_no_mask']
        best_dice_with_no_mask = state['best_dice_with_no_mask']

        print(epoch, best_thr_with_no_mask, best_dice_with_no_mask)
        with open(checkpoint_path + '/swa.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([epoch, best_thr_with_no_mask, best_dice_with_no_mask])          
        for i, ckpt in enumerate(checkpoints[1:]):
            model_2 = eval(model_name+'()')
            model_2 = torch.nn.DataParallel(model_2).cuda()
            state = torch.load(ckpt)
            epoch = state['epoch']
            best_thr_with_no_mask = state['best_thr_with_no_mask']
            best_dice_with_no_mask = state['best_dice_with_no_mask']

            print(epoch, best_thr_with_no_mask, best_dice_with_no_mask)
            with open(checkpoint_path + '/swa.csv', 'a', newline='') as f:
                writer = csv.writer(f)

                writer.writerow([epoch, best_thr_with_no_mask, best_dice_with_no_mask])             
            model_2.load_state_dict(state['state_dict'])                      
            moving_average(model_1, model_2, 1. / (i + 2))

        with torch.no_grad():
            bn_update(train_loader, model_1)
            valLoss, valLoss_seg, valLoss_cls, auc, max_threshold_list, max_result_f1_list, precision_list, recall_list = epochVal(num_fold, model_1, val_loader, loss_seg, loss_cls, c_val, val_batch_size, cls_weight, seg_weight)

        result = ['swa', round(np.mean(auc), 3)]

        print(result)
        
        torch.save({'state_dict': model_1.state_dict()}, checkpoint_path + '/model_epoch_swa_valid_transform_aug_' + str(num_fold) + '.pth')
        
        with open(checkpoint_path + '/swa.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(result)  


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

    parser.add_argument("-cp", "--checkpoint_path", type=str,
                        default='diy_model_se_resnext50_32x4d_768', help='epoch')

    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 16
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)

    checkpoint_path = path_data['snapshot_path'] + args.checkpoint_path.replace('\n', '').replace('\r', '')
    epoch_list = [24, 29, 30, 31, 32, 33]
    backbone = args.backbone
    print(backbone)
    train_one_model(backbone, Image_size)
