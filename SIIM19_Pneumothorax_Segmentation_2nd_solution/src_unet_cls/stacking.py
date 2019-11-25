import gc
import numpy as np  # linear algebra
import pandas as pd  # data processing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import fbeta_score
import os
from copy import deepcopy
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.ranking import roc_auc_score
from glob import glob

from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def computeAUROC(dataGT, dataPRED, classCount):

    outAUROC = []

    datanpGT = dataGT
    datanpPRED = dataPRED
    for i in range(classCount):
        auc = roc_auc_score(datanpGT[:, i], datanpPRED[:, i])
        outAUROC.append(round(auc, 3))

    return outAUROC

def f1_score_mine(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return f1_score(y_true, y_pred)

def search_f1(output, target):

    output_class = output
    target_class = target
    max_result_f1 = 0
    max_threshold = 0
    for threshold in [x * 0.01 for x in range(0, 100)]:
        prob = output_class > threshold
        label = target_class
        result_f1 = f1_score_mine(label, prob)
        if result_f1 > max_result_f1:
            max_result_f1 = result_f1
            max_threshold = threshold

    return max_threshold, round(max_result_f1,3)

def compute_metric(predict, coupling_value):
    predict = predict.reshape([-1])
    coupling_value = coupling_value.reshape([-1])

    diff = np.fabs(predict - coupling_value)
    m = np.mean(diff)
    log_m = np.log(m)
    return log_m


class StackingDataset(Dataset):
    def __init__(self, imagelist, labellist=None, model_num = None, mode='train'):
        self.imagelist = imagelist
        self.labellist = labellist
        self.mode = mode
        self.model_num = model_num

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode == 'valid':
            x = deepcopy(self.imagelist[idx])
            x = x.reshape(self.model_num, -1)
            x = torch.from_numpy(x).unsqueeze(0)
            y = self.labellist[idx]
            return x, y
        else:
            x = deepcopy(self.imagelist[idx])
            x = x.reshape(self.model_num, -1)
            x = torch.from_numpy(x).unsqueeze(0)
            return x

class Simple(nn.Module):
    def __init__(self, model_num):
        super(Simple, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(model_num, 1), stride=(model_num,1), bias=True))
        self.last_linear = nn.Sequential(nn.Linear(16, 1))

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.layer1(x)
        x = x.view(batch_size, -1)
        x = self.last_linear(x)
        return x

def move(lst, k):
    return lst[k:] + lst[:k]

def get_X(x_list):
    X = []
    X += x_list
    x_mean = np.mean(x_list,axis=0)
    X.append(x_mean)
    x_list_move = move(x_list, 1)
    for x0, x1 in zip(x_list, x_list_move):
        X.append((x0-x1)/2.0)

    return X


f1 = open('../configs/seg_path_configs.json', encoding='utf-8')
path_data = json.load(f1)
model_save_dir = path_data['stacking_path']
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
model_dirs = glob(path_data['snapshot_path'] + '*/prediction_swa/*')
model_dirs = [tmp for tmp in model_dirs if 'val_oof' in tmp]
print(model_dirs)
X = []
for model_dir in model_dirs:
    print(model_dir)
    model_csv = pd.read_csv(model_dir)
    x = np.asarray(model_csv.sort_values(by='ImageId')['preds'].values).astype(np.float32).reshape([-1, 1])
    y = np.asarray(model_csv.sort_values(by='ImageId').reset_index(drop = True).loc[:, 'gt'].values).astype(np.float32)
    print(search_f1(x, y))
    X.append(x)
    
print(search_f1(np.mean(X,axis=0), y))
X = get_X(X)
model_num = len(X)
X = np.concatenate(X,axis=1).astype(np.float32)
run = True
criterion = torch.nn.BCEWithLogitsLoss()

if run:
    kf = KFold(n_splits=5, shuffle=True, random_state=48)
    for s_fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # if fold == s_fold:
        x_train = X[train_idx]
        y_train = y[train_idx]
        # print(x_train)
        x_val = X[val_idx]
        y_val = y[val_idx]
        print('fold ' + str(s_fold))
        batch_size = 128
        train_data = StackingDataset(x_train, y_train, model_num = model_num)
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, pin_memory=True, drop_last=True, shuffle=True)
        val_data = StackingDataset(x_val, y_val, model_num = model_num)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=True, drop_last=False, shuffle=False)
        model = Simple(model_num = model_num).cuda()
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], gamma=0.1)

        best_score = 100
        for epoch in range(50):
            running_loss = 0.0
            model.train()
            for data, labels in tqdm(train_loader, position=0):
                data, labels = data.float().cuda(), labels.float().cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    data, labels = data.cuda(), labels.cuda()
                    labels = labels.view(-1, 1).contiguous().cuda(async=True)
                    logit = model(data)
                    
                    loss = criterion(logit, labels)
                    logit = logit.sigmoid()
                    loss.backward()
                    optimizer.step()
                running_loss += loss * data.shape[0]
            train_loss = running_loss / train_data.__len__()
            scheduler.step()

            running_loss = 0
            count = 0
            model.eval()
            for inputs, labels in tqdm(val_loader, position=0):
                inputs, labels = inputs.float().cuda(), labels.float().cuda()
                labels = labels.view(-1, 1).contiguous().cuda(async=True)

                with torch.set_grad_enabled(False):
                    logit = model(inputs)
                    loss = criterion(logit, labels)
                    logit = logit.sigmoid()
                if count != 0:
                    predicts = np.vstack((predicts, logit.cpu().numpy()))
                    truths = np.vstack((truths, labels.cpu().numpy().reshape([-1,1])))
                else:
                    predicts = logit.cpu().numpy()
                    truths = labels.cpu().numpy().reshape([-1,1])

                count += 1
                running_loss += loss.item() * inputs.size(0)

            logMAE = search_f1(predicts, truths)
            auc = computeAUROC(truths, predicts, 1)
            val_loss = running_loss / val_data.__len__()
            print(str(epoch), 'train_loss:{} val_loss:{} LogMAE:score:{} auc:{}'.format(train_loss, val_loss, logMAE, auc))
            if best_score > val_loss:
                best_score = val_loss
                print('save max score!!!!!!!!!!!!')
                torch.save(model.state_dict(), os.path.join(model_save_dir,'type_' + str(s_fold) + '.pt'))
            del predicts, truths
            gc.collect()

if run:
    count = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=48)
    for s_fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        x_train = X[train_idx]
        y_train = y[train_idx]

        x_val = X[val_idx]
        y_val = y[val_idx]

        batch_size = 128
        train_data = StackingDataset(x_train, y_train, model_num = model_num)
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, pin_memory=True, drop_last=True, shuffle=True)
        val_data = StackingDataset(x_val, y_val, model_num = model_num)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=True, drop_last=False, shuffle=False)
        model = Simple(model_num).cuda()

        model.load_state_dict(torch.load(os.path.join(model_save_dir,'type_' + str(s_fold) + '.pt')))
        running_loss = 0

        model.eval()
        for inputs, labels in tqdm(val_loader, position=0):
            inputs, labels = inputs.float().cuda(), labels.float().cuda()
            labels = labels.view(-1, 1).contiguous().cuda(async=True)

            with torch.set_grad_enabled(False):
                logit = model(inputs)

                loss = criterion(logit, labels)
                logit = logit.sigmoid()
            if count != 0:
                predicts = np.vstack((predicts, logit.cpu().numpy()))
                truths = np.vstack((truths, labels.cpu().numpy().reshape([-1, 1])))
            else:
                predicts = logit.cpu().numpy()
                truths = labels.cpu().numpy().reshape([-1, 1])

            count += 1
            running_loss += loss.item() * inputs.size(0)

        print('===============================================================================================')

    logMAE = search_f1(predicts, truths)
    auc = computeAUROC(truths, predicts, 1)
    print(running_loss / val_data.__len__(), logMAE, auc)


if run:
    batch_size = 128
    subs = [ tmp.replace('val_oof', 'test_pred') for tmp in model_dirs]

    x = []
    for sub in subs:

        sub_tmp = pd.read_csv(sub)
        x_tmp = np.asarray(sub_tmp.sort_values(by='ImageId')['preds'].values).astype(np.float32).reshape([-1,1])
        x.append(x_tmp)
        ids = sub_tmp.sort_values(by='ImageId')['ImageId'].values.tolist()

    x = get_X(x)
    x = np.concatenate(x, axis=1).astype(np.float32)

    sub_predicts = []
    for s_fold in range(5):
        model = Simple(model_num).cuda()
        print(os.path.join(model_save_dir,'type_' + str(s_fold) + '.pt'))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'type_' + str(s_fold) + '.pt')))
        model.eval()

        infer_data = StackingDataset(x, None,model_num = model_num , mode='test')
        infer_loader = DataLoader(infer_data, batch_size=batch_size,
                                  num_workers=2,pin_memory=True,
                                  drop_last=False, shuffle=False)

        count = 0
        for inputs in tqdm(infer_loader, position=0):
            inputs = inputs.float().cuda()

            with torch.set_grad_enabled(False):
                logit = model(inputs)
                logit = logit.sigmoid()

            if count != 0:
                predicts = np.vstack((predicts, logit.cpu().numpy()))
            else:
                predicts = logit.cpu().numpy()
            count += 1

        sub_predicts.append(predicts)
        print('===============================================================================================')

    print(len(sub_predicts))
    print(np.mean(sub_predicts,axis=0).shape)
    csv = os.path.join(model_save_dir, 'stacking_preds.csv')
    pd.DataFrame({'ImageId': list(ids), 'preds': list(np.mean(sub_predicts,axis=0).reshape([-1]))}).to_csv(csv,index=None)

df1 = pd.read_csv(csv)
df2 = pd.read_csv('../result/ensemble_5fold_0.4.csv')
df_old = pd.read_csv('../data/competition_data/sample_submission.csv')
all_name = df1['ImageId'].tolist()
all_result = []
for name in all_name:
    
    if df_old[df_old['ImageId'] == name].shape[0] > 1 or df1[df1['ImageId'] == name]['preds'].values[0] > 0.65:
#         print(df1[df1['ImageId'] == name]['preds'].values[0])
        all_result.append(df2[df2['ImageId'] == name]['EncodedPixels'].values[0])

    else:
        all_result.append('-1')
#     print(all_result)
d = {'ImageId': all_name, 'EncodedPixels': all_result}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(model_save_dir, 'submission_swa_0.65.csv'), index=False)