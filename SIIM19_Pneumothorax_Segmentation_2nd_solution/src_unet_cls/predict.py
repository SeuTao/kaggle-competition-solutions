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
import sys
lib_path = os.path.abspath('/data/VPS/VPS_04/kaggle/kaggle_siim/src_unet_cls')
sys.path.append(lib_path) 
from models.model_unet import *
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW
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
torch.backends.cudnn.benchmark = True
from multiprocessing import Pool
import PIL
import sklearn
import argparse



class Siim_Dataset_test(data.Dataset):

    def __init__(self,
                 name_list = None,
                 transform = None
                 ):
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx]
        image = cv2.imread('/home1/kaggle_siim/process/test/' + name + '.png')

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image'].transpose(2, 0, 1)

        return image

def predict(model_name, image_size):

    test_transform = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])

    c_test = sample_submission['ImageId'].tolist()

    test_dataset_512 = Siim_Dataset_test(c_test, test_transform)

    test_loader_512 = torch.utils.data.DataLoader(
        test_dataset_512,
        batch_size=val_batch_size,        
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    PRED_5_FOLD_cls_test = []
    PRED_5_FOLD_cls_val = []
    GT_5_FOLD_cls_val = []
    val_name_list = []


    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)


    for num_fold in range(5):
        print(num_fold)
        # if num_fold in [0]:
        #     continue

        f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]  
        
        val_name_list = val_name_list + c_val
        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = eval(model_name+'()')
        model = torch.nn.DataParallel(model).cuda()
    #     state = torch.load(snapshot_path + '/model_epoch_33_' + str(num_fold) + '.pth')
        state = torch.load(snapshot_path + '/model_epoch_swa_valid_transform_aug_' + str(num_fold) + '.pth')
        model.load_state_dict(state['state_dict'])
        model.eval ()

        outPRED_cls_val = torch.FloatTensor().cuda()
        outGT_cls_val = torch.FloatTensor().cuda()
        outPRED = []
        outGT = []

        for i, (input, target_seg, target_cls) in enumerate (val_loader):

            if i == 0:
                ss_time = time.time()
            print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
            varInput = torch.autograd.Variable(input)
            target_cls = target_cls.view(-1, 1).contiguous().cuda(async=True)
            outGT_cls_val = torch.cat((outGT_cls_val, target_cls), 0)
            outGT.append(target_seg.data.cpu().float())
            varOutput_cls, varOutput_seg = model(varInput)
            varOutput_seg = varOutput_seg.sigmoid()
            varOutput_cls = varOutput_cls.sigmoid()
            outPRED_cls_val = torch.cat((outPRED_cls_val, varOutput_cls.data), 0)
            outPRED.append(varOutput_seg.data.cpu().float())
            del varOutput_cls, varOutput_seg
            
        PRED_5_FOLD_cls_val.append(outPRED_cls_val)
        GT_5_FOLD_cls_val.append(outGT_cls_val)
        auc = computeAUROC(outGT_cls_val, outPRED_cls_val, 1)
        print('auc:', auc)
         
        outPRED_cls_test = torch.FloatTensor()
        for i, input in enumerate (test_loader_512):

            if i == 0:
                ss_time = time.time()
            print(str(i) + '/' + str(int(len(c_test)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
            with torch.no_grad():
                varInput = torch.autograd.Variable(input)
            varOutput_cls, varOutput = model(varInput)
            varOutput = varOutput.sigmoid()
            varOutput_cls = varOutput_cls.sigmoid()
            outPRED_cls_test = torch.cat((outPRED_cls_test, varOutput_cls.data.cpu()), 0)
            del varOutput_cls, varOutput

        PRED_5_FOLD_cls_test.append(outPRED_cls_test)
        
    pred_val = torch.cat(PRED_5_FOLD_cls_val).cpu().numpy().ravel()
    gt_val = torch.cat(GT_5_FOLD_cls_val).cpu().numpy().ravel()
    val_name_list = [x.replace('.png', '') for x in val_name_list]
    d_val = {'ImageId': val_name_list, 'preds': pred_val, 'gt': gt_val}
    df_val = pd.DataFrame(data=d_val)
    df_val.to_csv(prediction_path + 'val_oof_' + args.snapshot_path.replace('\n', '').replace('\r', '') + '_swa.csv', index=False)

    PRED_5_FOLD_cls_test = [x.numpy().ravel() for x in PRED_5_FOLD_cls_test]
    pred_test = (PRED_5_FOLD_cls_test[0] + PRED_5_FOLD_cls_test[1] + PRED_5_FOLD_cls_test[2] + PRED_5_FOLD_cls_test[3] + PRED_5_FOLD_cls_test[4])/5
    c_test = [x.replace('.png', '') for x in c_test]
    d_test = {'ImageId': c_test, 'preds': pred_test}
    df_test = pd.DataFrame(data=d_test)
    df_test.to_csv(prediction_path + 'test_pred_' + args.snapshot_path.replace('\n', '').replace('\r', '') + '_swa.csv', index=False)

    df1 = pd.read_csv(prediction_path + 'test_pred_' + args.snapshot_path.replace('\n', '').replace('\r', '') + '_swa.csv')
    df2 = pd.read_csv('../result/ensemble_5fold_0.4.csv')
    df_old = pd.read_csv(path_data['sample_submission'])
    all_name = df1['ImageId'].tolist()
    all_result = []
    for name in all_name:
        
        if df_old[df_old['ImageId'] == name].shape[0] > 1 or df1[df1['ImageId'] == name]['preds'].values[0] > 0.7:
            all_result.append(df2[df2['ImageId'] == name]['EncodedPixels'].values[0])
        else:
            all_result.append('-1')

    d = {'ImageId': all_name, 'EncodedPixels': all_result}
    df = pd.DataFrame(data=d)
    df.to_csv(prediction_path + 'submission_swa_0.7.csv', index=False)


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

    parser.add_argument("-spth", "--snapshot_path", type=str,
                        default='diy_model_se_resnext50_32x4d_768_normal', help='epoch')

    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 4
    print(Image_size)
    print(train_batch_size)
    print(val_batch_size)

    snapshot_path = path_data['snapshot_path'] + args.snapshot_path.replace('\n', '').replace('\r', '')
    kfold_path = path_data['kfold_path_cls']
    sample_submission = pd.read_csv(path_data['sample_submission'])
    df_all = pd.read_csv(csv_path)
    prediction_path = snapshot_path + '/prediction_swa/'

    backbone = args.backbone
    print(backbone)
    predict(backbone, Image_size)
