# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     nn_submit
   Description :
   Author :       haxu
   date：          2019-06-27
-------------------------------------------------
   Change Activity:
                   2019-06-27:
-------------------------------------------------
"""
__author__ = 'haxu'

import torch
import pandas as pd
from sklearn.model_selection import KFold
from nn_utils import Graph, Coupling, compute_kaggle_metric, do_valid
from nn_data import PMPDataset, DataLoader, null_collate
from model import Net
from tqdm import tqdm
import numpy as np
import random

device = torch.device('cuda:6')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def submit_fold(names, net, device=device):
    loader = DataLoader(PMPDataset(names), batch_size=128, collate_fn=null_collate, num_workers=8, pin_memory=True)
    net.eval()
    test_num = 0
    test_predict = []
    test_coupling_value = []
    test_id = []

    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in tqdm(enumerate(loader)):
        with torch.no_grad():
            node = node.to(device)
            edge = edge.to(device)
            edge_index = edge_index.to(device)
            node_index = node_index.to(device)
            coupling_value = coupling_value.to(device)
            coupling_index = coupling_index.to(device)

            predict = net(node, edge, edge_index, node_index, coupling_index)

        batch_size = len(infor)
        test_id.extend(list(np.concatenate([infor[b][2] for b in range(batch_size)])))

        test_predict.append(predict.data.cpu().numpy())
        test_coupling_value.append(coupling_value.data.cpu().numpy())
        test_num += batch_size

    id = test_id
    predict = np.concatenate(test_predict)

    assert len(id) == len(predict)

    return id, predict, np.concatenate(test_coupling_value)


def submit_all_fold():
    ids = 0
    res = 0

    df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv', usecols=['molecule_name'])
    test_names = df_test['molecule_name'].unique()

    for fold in range(5):
        print(f'fold{fold}.........')
        net = Net().to(device)

        net.load_state_dict(
            torch.load(f'../checkpoint/fold{fold}_model_0811-fine.pth', map_location=lambda storage, loc: storage))
        print('load pre-trained done........')
        id, predict, coupling_value = submit_fold(test_names, net)
        ids = id
        res += predict / 5

    df = pd.DataFrame(list(zip(ids, res)), columns=['id', 'scalar_coupling_constant'])
    df.to_csv('submission_5fold.csv', index=False)


def submit_one_fold():
    print('submit 0 fold........')
    df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv', usecols=['molecule_name'])
    test_names = df_test['molecule_name'].unique()
    net = Net().to(device)
    net.load_state_dict(
        torch.load(f'../checkpoint/fold0_model_0808_new.pth', map_location=lambda storage, loc: storage))
    id, predict, coupling_value = submit_fold(test_names, net)
    df = pd.DataFrame(list(zip(id, predict)), columns=['id', 'scalar_coupling_constant'])
    df.to_csv('submission_0.csv', index=False)


def get_cv_score():
    names = np.load('../input/champs-scalar-coupling/names.npy')

    net = Net().to(device)
    cv_score = []
    print('1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH')
    for k in range(5):
        # if k != 0:
        #     continue
        net.load_state_dict(
            torch.load(f'../checkpoint/fold{k}_model_0823.pth', map_location=lambda storage, loc: storage))

        loader = DataLoader(PMPDataset(names[k]), batch_size=128, collate_fn=null_collate, num_workers=8,
                            pin_memory=False)
        _, log_mae, log_mae_mean = do_valid(net, loader, device)
        cv_score.append(log_mae_mean)
        print('{:^7.4f}, {:^7.4f}, {:^7.4f}, {:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f}'.format(*log_mae))
        print(f'fold{k} score {log_mae_mean}..')

    print(f'cv score {np.mean(cv_score)}..........')


norm = {
    '1JHC': [94.976153, 18.277236880290022],
    '2JHH': [-10.286605, 3.979607163730381],
    '1JHN': [47.479884, 10.922171556272295],
    '2JHN': [3.124754, 3.6734741723096236],
    '2JHC': [-0.270624, 4.523610750196486],
    '3JHH': [4.771023, 3.704984434128579],
    '3JHC': [3.688470, 3.0709074866562176],
    '3JHN': [0.990730, 1.315393353533751],
}

if __name__ == '__main__':
    get_cv_score()
    submit_one_fold()
    submit_all_fold()
