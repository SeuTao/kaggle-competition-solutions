import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import Net
from nn_utils import do_train, do_valid, time_to_str, Graph, Coupling, Logger, adjust_learning_rate
from Nadam import Nadam
from timeit import default_timer as timer
from nn_data import PMPDataset, null_collate
from argparse import ArgumentParser
import random
from itertools import chain
from tqdm import tqdm
import pickle

parser = ArgumentParser(description='train PMP')

parser.add_argument('--fold', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--bs', type=int, default=48)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()

fold = args.fold
lr = args.lr
bs = args.bs
name = args.name

device = torch.device(args.gpu)
cuda_aviable = torch.cuda.is_available()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if cuda_aviable:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

log = Logger()
log.open(f'{name}_fold{fold}.txt')
log.write(str(args) + '\n')


def select(names):
    res = []

    for x in tqdm(names):
        with open(f'../input/graph0821/{x}.pickle', 'rb') as f:
            g = pickle.load(f)
            assert isinstance(g, Graph)

            if len(g.axyz[1]) < 15:
                res.append(x)
    return res


def train_fold(fold):
    names = np.load('../input/champs-scalar-coupling/names.npy')

    for k in range(5):
        if k != fold:
            continue
        log.write(f'~~~~~~~~~~~~ fold {fold} ~~~~~~~~~~~~\n')
        best_score = 999

        val_names = names[k]
        tr_names = list(chain(*(names[:k].tolist() + names[k + 1:].tolist())))

        print(len(tr_names), len(val_names))

        val_names = select(val_names)
        tr_names = select(tr_names)

        print(len(tr_names), len(val_names))

        train_loader = DataLoader(PMPDataset(tr_names), batch_size=bs, collate_fn=null_collate, num_workers=8,
                                  pin_memory=False,
                                  shuffle=True)
        val_loader = DataLoader(PMPDataset(val_names), batch_size=128, collate_fn=null_collate, num_workers=8,
                                pin_memory=False,
                                )

        net = Net().to(device)

        optimizer = Nadam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        if 'fine' in name:
            net.load_state_dict(
                torch.load(f'../checkpoint/fold{fold}_model_{name.split("-")[0]}.pth',
                           map_location=lambda storage, loc: storage))

            log.write('load pre-trained done.....\n')

        f_normal = '{:^5} | {:^3.4f} | {:^7.4f} | {:^7.4f}, {:^7.4f}, {:^7.4f}, {:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f} | {:^7.4f} | {:^7} \n'
        f_boost = '{:^5}* | {:^3.4f} | {:^7.4f} | {:^7.4f}, {:^7.4f}, {:^7.4f}, {:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f} | {:^7.4f} | {:^7} \n'

        log.write(
            'epoch | train loss |  valid loss |  1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH  |  log_mae  |  time \n')

        start = timer()
        for e in range(200):
            train_loss = do_train(net, train_loader, optimizer, device)

            valid_loss, log_mae, log_mae_mean = do_valid(net, val_loader, device)

            timing = time_to_str((timer() - start), 'min')

            if log_mae_mean < best_score:
                best_score = log_mae_mean
                torch.save(net.state_dict(), f'../checkpoint/fold{k}_model_{name}.pth')
                log.write(f_boost.format(e, train_loss, valid_loss, *log_mae, log_mae_mean, timing))

            else:
                log.write(f_normal.format(e, train_loss, valid_loss, *log_mae, log_mae_mean, timing))


if __name__ == '__main__':
    train_fold(fold)
