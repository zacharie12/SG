from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, GMMConv, GlobalAttention, global_mean_pool
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
        self.pooling = GlobalAttention(
            nn.Sequential(
                nn.Linear(in_channels, 1)
            ),
            nn.Sequential(
                nn.Linear(in_channels, in_channels)
            )
        )

    def forward(self, sg):
        x, edge_idx, edge_w = sg
        x = x.float()
        edge_w = edge_w.float()
        N = len(x)
        x = self.conv1(x, edge_idx, edge_w)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx, edge_w)
        device = x.get_device()
        batch = torch.zeros((N), dtype=torch.long, device=device)
        x = self.pooling(x, batch)
        return x

class GaussGNN(Gnn):
    def __init__(self, in_channels, dim, out_size):
        super(GaussGNN, self).__init__()
        self.conv1 = GMMConv(in_channels, in_channels,  dim, kernel_size=25)
        self.conv2 = GMMConv(in_channels, in_channels,  dim, kernel_size=25)
        self.conv3 = GMMConv(in_channels, in_channels,  dim, kernel_size=25)
        self.linear = nn.Linear(in_channels, in_channels)
        '''
        self.pooling = GlobalAttention(
            nn.Sequential(
                nn.Linear(in_channels, 1),
            ),
            nn.Sequential(
                nn.Linear(in_channels, in_channels)
            )
        )
        '''

    def forward(self, sg):
        x, edge_idx, edge_w = sg
        # x = x.float()
        # edge_w = edge_w.float()
        N = len(x)
        x = self.conv1(x, edge_idx, edge_w)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx, edge_w)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_idx, edge_w)
        x = self.linear(x)
        
        device = x.get_device()
        batch = torch.zeros((N), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        return x


GNN_ARCHITECHTURE = {'SimpleGNN':SimpleGNN, 'GaussGNN':GaussGNN }
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







