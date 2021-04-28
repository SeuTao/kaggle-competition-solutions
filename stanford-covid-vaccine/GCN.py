import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import json
import seaborn as sns
import os
import random
from tqdm.notebook import tqdm
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torch.nn import functional as F
from torch.optim import lr_scheduler

class config:
    train_file = './stanford-covid-vaccine/train.json'
    test_file = './stanford-covid-vaccine/test.json'
    pretrain_dir = './'
    sample_submission = './stanford-covid-vaccine/sample_submission.csv'
    learning_rate = 0.001
    batch_size = 64
    n_epoch = 100
    n_split = 5
    K = 1 # number of aggregation loop (also means number of GCN layers)
    gcn_agg = 'mean' # aggregator function: mean, conv, lstm, pooling
    filter_noise = True
    seed = 1234

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(config.seed)

class GCN(nn.Module):
    '''
    Implementation of one layer of GraphSAGE
    '''
    def __init__(self, input_dim, output_dim, aggregator='mean'):
        super(GCN, self).__init__()
        self.aggregator = aggregator

        if aggregator == 'mean':
            linear_input_dim = input_dim * 2
        elif aggregator == 'conv':
            linear_input_dim = input_dim
        elif aggregator == 'pooling':
            linear_input_dim = input_dim * 2
            self.linear_pooling = nn.Linear(input_dim, input_dim)
        elif aggregator == 'lstm':
            self.lstm_hidden = 128
            linear_input_dim = input_dim + self.lstm_hidden
            self.lstm_agg = nn.LSTM(input_dim, self.lstm_hidden, num_layers=1, batch_first=True)

        self.linear_gcn = nn.Linear(in_features=linear_input_dim, out_features=output_dim)

    def forward(self, input_, adj_matrix):
        if self.aggregator == 'conv':
            # set elements in diagonal of adj matrix to 1 with conv aggregator
            idx = torch.arange(0, adj_matrix.shape[-1], out=torch.LongTensor())
            adj_matrix[:, idx, idx] = 1

        adj_matrix = adj_matrix.type(torch.float32)
        sum_adj = torch.sum(adj_matrix, axis=2)
        sum_adj[sum_adj == 0] = 1

        if self.aggregator == 'mean' or self.aggregator == 'conv':
            feature_agg = torch.bmm(adj_matrix, input_)
            feature_agg = feature_agg / sum_adj.unsqueeze(dim=2)

        elif self.aggregator == 'pooling':
            feature_pooling = self.linear_pooling(input_)
            feature_agg = torch.sigmoid(feature_pooling)
            feature_agg = torch.bmm(adj_matrix, feature_agg)
            feature_agg = feature_agg / sum_adj.unsqueeze(dim=2)

        elif self.aggregator == 'lstm':
            feature_agg = torch.zeros(input_.shape[0], input_.shape[1], self.lstm_hidden).cuda()
            for i in range(adj_matrix.shape[1]):
                neighbors = adj_matrix[:, i, :].unsqueeze(2) * input_
                _, hn = self.lstm_agg(neighbors)
                feature_agg[:, i, :] = torch.squeeze(hn[0], 0)

        if self.aggregator != 'conv':
            feature_cat = torch.cat((input_, feature_agg), axis=2)
        else:
            feature_cat = feature_agg

        feature = torch.sigmoid(self.linear_gcn(feature_cat))
        feature = feature / torch.norm(feature, p=2, dim=2).unsqueeze(dim=2)

        return feature

class Net(nn.Module):
    def __init__(self, num_embedding=14, seq_len=107, pred_len=68, dropout=0.5,
                 embed_dim=100, hidden_dim=128, K=1, aggregator='mean'):
        '''
        K: number of GCN layers
        aggregator: type of aggregator function
        '''
        super(Net, self).__init__()

        self.pred_len = pred_len
        self.embedding_layer = nn.Embedding(num_embeddings=num_embedding,
                                            embedding_dim=embed_dim)

        self.gcn = nn.ModuleList([GCN(3 * embed_dim, 3 * embed_dim, aggregator=aggregator) for i in range(K)])

        self.gru_layer = nn.GRU(input_size=3 * embed_dim,
                                hidden_size=hidden_dim,
                                num_layers=3,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)

        self.linear_layer = nn.Linear(in_features=2 * hidden_dim,
                                      out_features=5)

    def forward(self, input_, adj_matrix):
        # embedding
        embedding = self.embedding_layer(input_)
        embedding = torch.reshape(embedding, (-1, embedding.shape[1], embedding.shape[2] * embedding.shape[3]))
        # gcn
        gcn_feature = embedding
        for gcn_layer in self.gcn:
            gcn_feature = gcn_layer(gcn_feature, adj_matrix)

        # gru
        gru_output, gru_hidden = self.gru_layer(gcn_feature)
        truncated = gru_output[:, :self.pred_len]

        output = self.linear_layer(truncated)
        return output

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}


def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)
    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])

    assert len(couples) == len(opened)

    return couples


def build_matrix(couples, size):
    mat = np.zeros((size, size))

    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1

    for i, j in couples:
        mat[i, j] = 1
        mat[j, i] = 1

    return mat


def convert_to_adj(structure):
    couples = get_couples(structure)
    mat = build_matrix(couples, len(structure))
    return mat


def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    inputs = np.transpose(
        np.array(
            df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
        ),
        (0, 2, 1)
    )

    adj_matrix = np.array(df['structure'].apply(convert_to_adj).values.tolist())

    return inputs, adj_matrix


train = pd.read_json(config.train_file, lines=True)

if config.filter_noise:
    train = train[train.signal_to_noise > 1]

test = pd.read_json(config.test_file, lines=True)
sample_df = pd.read_csv(config.sample_submission)

# %%

train_inputs, train_adj = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))

train_inputs = torch.tensor(train_inputs, dtype=torch.long)
train_adj = torch.tensor(train_adj, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.float32)


def train_fn(epoch, model, train_loader, criterion, optimizer):
    model.train()
    model.zero_grad()
    train_loss = AverageMeter()

    for index, (input_, adj, label) in enumerate(train_loader):
        input_ = input_.cuda()
        adj = adj.cuda()
        label = label.cuda()
        preds = model(input_, adj)

        loss = criterion(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item())

    print(f"Train loss {train_loss.avg}")
    return train_loss.avg


def eval_fn(epoch, model, valid_loader, criterion):
    model.eval()
    eval_loss = AverageMeter()

    for index, (input_, adj, label) in enumerate(valid_loader):
        input_ = input_.cuda()
        adj = adj.cuda()
        label = label.cuda()
        preds = model(input_, adj)

        loss = criterion(preds, label)
        eval_loss.update(loss.item())

    print(f"Valid loss {eval_loss.avg}")
    return eval_loss.avg


# %%

def run(fold, train_loader, valid_loader):
    model = Net(K=config.K, aggregator=config.gcn_agg)
    model.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=0.0)

    train_losses = []
    eval_losses = []
    for epoch in range(config.n_epoch):
        print('#################')
        print('###Epoch:', epoch)

        train_loss = train_fn(epoch, model, train_loader, criterion, optimizer)
        eval_loss = eval_fn(epoch, model, valid_loader, criterion)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

    torch.save(model.state_dict(), f'{config.pretrain_dir}/gcn_gru_{fold}.pt')
    return train_losses, eval_losses


# %%

splits = KFold(n_splits=config.n_split, shuffle=True, random_state=config.seed).split(train_inputs)

for fold, (train_idx, val_idx) in enumerate(splits):
    train_dataset = TensorDataset(train_inputs[train_idx], train_adj[train_idx], train_labels[train_idx])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    valid_dataset = TensorDataset(train_inputs[val_idx], train_adj[val_idx], train_labels[val_idx])
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

    train_losses, eval_losses = run(fold, train_loader, valid_loader)

fig = px.line(
    pd.DataFrame([train_losses, eval_losses], index=['loss', 'val_loss']).T,
    y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'Mean Squared Error'},
    title='Training History')
fig.show()

#%%

public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs, public_adj = preprocess_inputs(public_df)
private_inputs, private_adj = preprocess_inputs(private_df)

public_inputs = torch.tensor(public_inputs, dtype=torch.long)
private_inputs = torch.tensor(private_inputs, dtype=torch.long)
public_adj = torch.tensor(public_adj, dtype=torch.long)
private_adj = torch.tensor(private_adj, dtype=torch.long)

model_short = Net(seq_len=107, pred_len=107, K=config.K, aggregator=config.gcn_agg)
model_long = Net(seq_len=130, pred_len=130, K=config.K, aggregator=config.gcn_agg)

list_public_preds = []
list_private_preds = []
for fold in range(config.n_split):
    model_short.load_state_dict(torch.load(f'{config.pretrain_dir}/gcn_gru_{fold}.pt'))
    model_long.load_state_dict(torch.load(f'{config.pretrain_dir}/gcn_gru_{fold}.pt'))
    model_short.cuda()
    model_long.cuda()
    model_short.eval()
    model_long.eval()

    public_preds = model_short(public_inputs.cuda(), public_adj.cuda())
    private_preds = model_long(private_inputs.cuda(), private_adj.cuda())
    public_preds = public_preds.cpu().detach().numpy()
    private_preds = private_preds.cpu().detach().numpy()

    list_public_preds.append(public_preds)
    list_private_preds.append(private_preds)

public_preds = np.mean(list_public_preds, axis=0)
private_preds = np.mean(list_private_preds, axis=0)

#%%
preds_ls = []
for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
#%%
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)

#%%
submission.head()