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




# Attention GNN
class AGNN_layer(nn.Module):
    def __init__(self, in_size, edge_size):
        super(AGNN_layer, self).__init__()

        self.in_size = in_size
        self.out_channel = in_size
        self.edge_size = edge_size

        self.att = nn.Sequential(
            nn.Linear(2 * in_size + edge_size, 2*in_size + edge_size),
            nn.ReLU(),
            nn.Linear(2 * in_size + edge_size, 1)
        )

        self.lin = nn.Sequential(
            nn.Linear(in_size, in_size)
        )

        self.gru = nn.GRUCell(in_size, in_size)

    def forward(self, x, edge_idx, edge_weights):
        num_nodes = x.size(0)

        dest_features_linearized = self.lin(x)

        new_x = []
        # for each node in the graph
        for node_idx in range(num_nodes):
            # find indices of all of this neighbors
            neighbor_indices = [edge_idx[1][t] for t in range(edge_idx.size(1)) if edge_idx[0][t] == node_idx]

            if len(neighbor_indices) == 0:
                new_x.append(x[node_idx])
                continue
            
            # tensor containing all indices of destinations going out of x
            dest_idx = torch.stack(neighbor_indices).type(torch.LongTensor).t()
            dest_idx = dest_idx.to(x.device)
            # list of indices of edges
            edge_indices = torch.stack([torch.IntTensor([t]) for t in range(edge_idx.size(1)) if edge_idx[0][t] == node_idx]).type(torch.LongTensor)
            edge_indices = edge_indices.squeeze()
            edge_indices = edge_indices.to(x.device)
            # matrix containing weights for all of the destinations
            dest_features = torch.index_select(x, 0, dest_idx)
            # matrix containing weights for all of these edges
            edges = torch.index_select(edge_weights, 0, edge_indices)
            
            X = x[node_idx].repeat(len(dest_idx)).view((dest_idx.size(0), self.in_size))
        
            concat_feat = torch.cat((X, edges, dest_features), dim=1)

            att_scores = self.att(concat_feat)
            att_logits = F.softmax(att_scores, dim=0)

            dest_features_lin = torch.index_select(dest_features_linearized, 0, dest_idx)

            hidden_t = (att_logits.view(-1, 1) * dest_features_lin).sum(dim=0).unsqueeze(dim=0)
            new_x.append(self.gru(x[node_idx].unsqueeze(dim=0), hidden_t).squeeze())
            
        return torch.stack(new_x)        

# Edge attention GNN
class EAGNN_layer(nn.Module):
    def __init__(self, in_size, edge_size):
        super(AGNN_layer, self).__init__()

        self.in_size = in_size
        self.out_channel = in_size
        self.edge_size = edge_size

        self.edge_lin = nn.Sequential(
            nn.Linear(2 * in_size + edge_size, 2*in_size + edge_size),
            nn.ReLU(),
            nn.Linear(2 * in_size + edge_size, edge_size)
        )
        self.att = nn.Sequential(
            nn.Linear(edge_size, edge_size),
            nn.ReLU(),
            nn.Linear(edge_size, 1)
        )

        self.lin = nn.Sequential(
            nn.Linear(in_size, in_size)
        )

        self.gru = nn.GRUCell(in_size, in_size)

    def forward(self, x, edge_idx, edge_weights):
        num_nodes = x.size(0)

        dest_features_linearized = self.lin(x)
        
        new_edge = torch.zeros(edge_weights.size())
        new_x = []
        # for each node in the graph
        for node_idx in range(num_nodes):
            # find indices of all of this neighbors
            neighbor_indices = [edge_idx[1][t] for t in range(edge_idx.size(1)) if edge_idx[0][t] == node_idx]

            if len(neighbor_indices) == 0:
                new_x.append(x[node_idx])
                continue
            
            # tensor containing all indices of destinations going out of x
            dest_idx = torch.stack(neighbor_indices).type(torch.LongTensor).t()
            dest_idx = dest_idx.to(x.device)
            # list of indices of edges
            edge_indices = torch.stack([torch.IntTensor([t]) for t in range(edge_idx.size(1)) if edge_idx[0][t] == node_idx]).type(torch.LongTensor)
            edge_indices = edge_indices.squeeze()
            edge_indices = edge_indices.to(x.device)
            # matrix containing weights for all of the destinations
            dest_features = torch.index_select(x, 0, dest_idx)
            # matrix containing weights for all of these edges
            edges = torch.index_select(edge_weights, 0, edge_indices)
            
            X = x[node_idx].repeat(len(dest_idx)).view((dest_idx.size(0), self.in_size))
        
            concat_feat = torch.cat((X, edges, dest_features), dim=1)

            att_scores = self.att(concat_feat)
            att_logits = F.softmax(att_scores, dim=0)

            dest_features_lin = torch.index_select(dest_features_linearized, 0, dest_idx)

            hidden_t = (att_logits.view(-1, 1) * dest_features_lin).sum(dim=0).unsqueeze(dim=0)
            new_x.append(self.gru(x[node_idx].unsqueeze(dim=0), hidden_t).squeeze())
            
        return torch.stack(new_x)  


class AGNN(Gnn):
    def __init__(self, in_channels, dim, out_size):
        super(AGNN, self).__init__()
        self.conv1 = AGNN_layer(in_channels, dim)
        self.conv2 = AGNN_layer(in_channels, dim)
        self.conv3 = AGNN_layer(in_channels, dim)
        self.linear = nn.Linear(in_channels, out_size)
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


GNN_ARCHITECHTURE = {'SimpleGNN':SimpleGNN, 'GaussGNN':GaussGNN, 'AGNN':AGNN }
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







