import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nn_data import PMPDataset, null_collate, Graph, Coupling
from torch_geometric.utils import scatter_
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import *
from torch.nn import Parameter
import math


class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-05, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class NodeEmbModule(nn.Module):
    def __init__(self, hidden):
        super(NodeEmbModule, self).__init__()
        self.fc1 = nn.Sequential(
            LinearBn(113, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            LinearBn(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        hidden_cat = hidden * 2
        self.se = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_cat // 4, hidden_cat),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            LinearBn(hidden_cat, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x = torch.cat([x1, x2], dim=-1)
        x = self.se(x) * x
        x = self.fc3(x)
        return x


class GCN(nn.Module):
    def __init__(self, node_dim, out_dim, dropout):
        super(GCN, self).__init__()

        self.in_channels = node_dim
        self.out_channels = out_dim
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(node_dim, out_dim))
        self.root_weight = Parameter(torch.Tensor(node_dim, out_dim))

        self.bias = Parameter(torch.Tensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        self.root_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        if edge_index.numel() > 0:
            row, col = edge_index

            out = torch.mm(x, self.weight)
            out_col = out[col]

            out_col = F.dropout(out_col, self.dropout, training=self.training)

            out = scatter_add(out_col, row, dim=0, dim_size=x.size(0))

            deg = scatter_add(
                x.new_ones((row.size())), row, dim=0, dim_size=x.size(0))
            out = out / deg.unsqueeze(-1).clamp(min=1)

            out = out + torch.mm(x, self.root_weight)
        else:
            out = torch.mm(x, self.root_weight)

        out = out + self.bias

        return out


class GraphConv(nn.Module):
    def __init__(self, node_dim, num_step):
        super(GraphConv, self).__init__()

        self.gru = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.edge_embedding = LinearBn(128, node_dim * node_dim)
        self.bias = nn.Parameter(torch.zeros(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))
        self.num_step = num_step
        self.node_dim = node_dim

    def forward(self, node, edge_index, edge):
        x = node
        hidden = node.unsqueeze(0)
        edge = self.edge_embedding(edge).view(-1, self.node_dim, self.node_dim)
        num_node, node_dim = node.shape  # 4,128
        nodes = []
        for i in range(self.num_step):
            nodes.append(node + x)

            x_i = torch.index_select(node, 0, edge_index[0])

            message = x_i.view(-1, 1, node_dim) @ edge  # 12,1,128

            message = message.view(-1, node_dim)  # 12,128

            message = scatter_('mean', message, edge_index[1], dim_size=num_node)

            message = F.relu(message + self.bias)  # 4, 128

            node = message  # 9, 128

            node, hidden = self.gru(node.view(1, -1, node_dim), hidden)  # (1,9,128)  (1,9,128)

            node = node.view(-1, node_dim)

        node = torch.stack(nodes)
        node = torch.mean(node, dim=0)
        return node


class Set2Set(torch.nn.Module):
    def softmax(self, x, index, num=None):
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layer = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1  # bs

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),  # hidden
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))  # cell

        q_star = x.new_zeros(batch_size, self.out_channel)  # bs,256
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)  # bs,128

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True)  # num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)  # num_node x 1

            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size)  #

            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr, batch):
        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)

        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)

        batch = batch[perm]

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        a = gmp(x, batch)
        m = gap(x, batch)

        return torch.cat([m, a], dim=1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden_dim = 128

        self.node_embedding = nn.Sequential(
            LinearBn(113, self.hidden_dim),
            nn.ReLU(inplace=True),
            LinearBn(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # self.node_embedding = NodeEmbModule(self.hidden_dim)
        self.edge_embedding = nn.Sequential(
            LinearBn(6, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True)
        )

        self.encoder1 = GraphConv(self.hidden_dim, 4)
        self.encoder2 = GCN(128, 128, dropout=0.1)
        self.encoder3 = GCN(128, 128, dropout=0.1)

        # self.decoder = Set2Set(self.hidden_dim, processing_step=4)

        self.decoder = SAGPool(self.hidden_dim)

        self.predict = nn.Sequential(
            LinearBn(6 * self.hidden_dim, 1024),
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 8),
        )

        self.node_se = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.Sigmoid()
        )
        self.node_down = nn.Sequential(
            LinearBn(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        edge_index = edge_index.t().contiguous()

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index, edge_coupling_index = \
            torch.split(coupling_index, 1, dim=1)

        node = self.node_embedding(node)
        edge = self.edge_embedding(edge)

        node = self.encoder1(node, edge_index, edge)

        node = self.encoder2(node, edge_index)

        node = self.encoder3(node, edge_index)

        pool = self.decoder(node, edge_index, edge, node_index)  # sagpool

        # pool = self.decoder(node, node_index)  # set2set

        pool = torch.index_select(pool, dim=0, index=coupling_batch_index.view(-1))  # 16,256
        node0 = torch.index_select(node, dim=0, index=coupling_atom0_index.view(-1))  # 16,128
        node1 = torch.index_select(node, dim=0, index=coupling_atom1_index.view(-1))  # 16,128
        edge = torch.index_select(edge, dim=0, index=edge_coupling_index.view(-1))

        att = node0 + node1 - node0 * node1

        predict = self.predict(torch.cat([pool, node0, node1, att, edge], -1))

        predict = torch.gather(predict, 1, coupling_type_index).view(-1)  # 16

        return predict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    names = ['dsgdb9nsd_000002', 'dsgdb9nsd_000001', 'dsgdb9nsd_000030', 'dsgdb9nsd_000038']
    train_loader = DataLoader(PMPDataset(names), batch_size=2, collate_fn=null_collate)
    net = Net()

    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(train_loader):
        _ = net(node, edge, edge_index, node_index, coupling_index)

        break

    print('model success!')
