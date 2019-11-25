# ============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.metrics.ranking import roc_auc_score

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torch.utils.data

import torch.utils.data as data
import sys

sys.path.insert(0, '..')
from src_unet.models.models import *
from src_unet.dataset.dataset import *
from src_unet.tuils.tools import *
from src_unet.tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW
from src_unet.tuils.loss_function import *
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

import PIL
import apex
from apex import amp


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


f1 = open('../configs/seg_path_configs.json', encoding='utf-8')
path_data = json.load(f1)
csv_path = path_data['train_rle_path']
test_img_path = path_data['test_img_path']

kfold_path = '../data/cls_fold_5_all_images/'
sample_submission = pd.read_csv('../data/competition_data/submission.csv')
result_save_path = '../result/'
Image_size = 1024
train_batch_size = 2
val_batch_size = 2
test_batch_size = 2
workers = 8

best_thr_r = 0.4


class Siim_Dataset_test(data.Dataset):

    def __init__(self,
                 name_list=None,
                 transform=None
                 ):
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image = cv2.imread(test_img_path + name + '.png')

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image'].transpose(2, 0, 1)

        return image


test_transform = albumentations.Compose([
    albumentations.Resize(768, 768, p=1),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

c_test = sample_submission['ImageId'].tolist()
test_dataset = Siim_Dataset_test(c_test, test_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
    drop_last=False)

PRED_5_FOLD = []

for num_fold in range(5):
    print('This is {} fold processing...'.format(num_fold))

    from ef_unet import EfficientNet_5_unet
    model = EfficientNet_5_unet()

    model = apex.parallel.convert_syncbn_model(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    model = torch.nn.DataParallel(model).cuda()
    state = torch.load('./ef5_swa_' + str(num_fold) + '.pth.tar')

    model.load_state_dict(state['state_dict'])

    model.eval()
    outPRED = []

    for i, input in enumerate(test_loader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_test) / val_batch_size)) + '     ' + str((time.time() - ss_time) / (i + 1)),
              end='\r')
        varInput = torch.autograd.Variable(input)
        varOutput = model(varInput)
        varOutput = F.upsample(varOutput, size=(1024, 1024), mode='bilinear')
        varOutput = varOutput.sigmoid()
        outPRED.append(varOutput.data.cpu().float())

    PRED_5_FOLD.append(outPRED)
PRED_5_FOLD = [torch.cat(x) for x in PRED_5_FOLD]
#
# ----------------------------------------------------------------
pred0 = (PRED_5_FOLD[0] + PRED_5_FOLD[1] + PRED_5_FOLD[2] + PRED_5_FOLD[3] + PRED_5_FOLD[4]) / 5
pred0 = pred0.numpy()
np.save('./ef5_pred.npy', pred0)