from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from math import sqrt


class Gnn(ABC, nn.Module):
    def __init__(self):
        super(Gnn, self).__init__()
        pass
    def forward(self, sg):
        pass


class SimpleGNN(Gnn):
    def __init__(self, in_channels, dim, out_size):
        super(SimpleGNN, self).__init__()
        self.conv1 = CGConv(in_channels, dim, aggr='add', bias=True)
        self.conv2 = CGConv(in_channels, dim,  aggr='add', bias=True)
        std_dev = float(1/sqrt(in_channels))
        self.node_w = nn.Linear(in_channels, out_size, bias=False)
        self.node_bias = nn.Parameter(torch.FloatTensor(out_size).uniform_(-std_dev, std_dev))

    def forward(self, sg):
        x, edge_idx, edge_w = sg
        x = self.conv1(x, edge_idx, edge_w)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx, edge_w)
        x = self.node_w(x)
        x = torch.sum(x, keepdim=True, dim=0)
        x += self.node_bias
        return x

GNN_ARCHITECHTURE = {'SimpleGNN':SimpleGNN }
def build_gnn(cfg):
    gnn = GNN_ARCHITECHTURE[cfg.LISTENER.GNN](cfg.LISTENER.NODE_SIZE, cfg.LISTENER.EDGE_SIZE, cfg.LISTENER.GNN_OUTPUT)
    return gnn

if __name__ == '__main__':
    gnn = SimpleGNN(1, 2, 4)
    x = torch.zeros((4, 1))
    y = torch.tensor([[0, 1, 1, 2],
                    [1, 0, 2, 3]], dtype=torch.long)
    w = torch.ones((4, 2))
    output = gnn((x, y, w))
    print(output)







