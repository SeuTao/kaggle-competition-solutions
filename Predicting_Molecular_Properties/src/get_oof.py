import torch
import pandas as pd
from sklearn.model_selection import KFold
from nn_utils import Graph, Coupling, compute_kaggle_metric, do_valid
from nn_data import PMPDataset, DataLoader, null_collate
from model import Net
from nn_submit import submit_fold
from tqdm import tqdm
import numpy as np
import random

device = torch.device('cuda:6')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log_mae(p, t):
    return np.log(np.fabs((p - t)).mean())


def get_valid_predict_type(type='1JHN'):
    train_names = np.load('../input/champs-scalar-coupling/names.npy')

    df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv', usecols=['id', 'molecule_name', 'type'])
    test_names = df_test['molecule_name'].unique()

    df = pd.read_csv('../input/champs-scalar-coupling/train.csv', usecols=['id', 'type'])

    net = Net().to(device)
    for k in range(5):
        print(f'{k}.....')
        net.load_state_dict(
            torch.load(f'../checkpoint/fold{k}_model_0823-fine.pth', map_location=lambda storage, loc: storage))

        id, predict, coupling_value = submit_fold(train_names[k], net, device=device)

        df.loc[df['id'].isin(id), 'scalar_coupling_constant'] = predict
        df.loc[df['id'].isin(id), 'label'] = coupling_value

        id, predict, coupling_value = submit_fold(test_names, net, device=device)

        df_test.loc[df_test['id'].isin(id), 'scalar_coupling_constant'] = predict
        df_test[df_test['type'] == type][['id', 'scalar_coupling_constant']].to_csv(f'submission_{type}_{k + 1}.csv',
                                                                                    index=False)

    df[df['type'] == type][['id', 'scalar_coupling_constant', 'label']].to_csv(f'oof_{type}_5fold.csv', index=False)


if __name__ == '__main__':
    for type in ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']:
        get_valid_predict_type(type=type)
