from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import CGConv, GMMConv, GlobalAttention, global_mean_pool,  MetaLayer
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
        self.conv3 = CGConv(in_channels, dim, aggr='add', bias=True)
        self.lin = nn.Linear(in_channels, out_size)

    def forward(self, sg):
        x, edge_idx, edge_w = sg
        x = x.float()
        edge_w = edge_w.float()
        N = len(x)
        x = self.conv1(x, edge_idx, edge_w)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx, edge_w)
        x = F.relu(x)
        x=self.conv3(x, edge_idx, edge_w)
        device = x.get_device()
        batch = torch.zeros((N), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class GaussGNN(Gnn):
    def __init__(self, in_channels, dim, out_size):
        super(GaussGNN, self).__init__()
        self.conv1 = GMMConv(in_channels, in_channels,  dim, kernel_size=25)
        self.conv2 = GMMConv(in_channels, in_channels,  dim, kernel_size=25)
        self.conv3 = GMMConv(in_channels, in_channels,  dim, kernel_size=25)
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
        
        device = x.get_device()
        batch = torch.zeros((N), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        x = self.linear(x)

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
            # calculae new features for edges
            edge_w = self.edge_lin(concat_feat)
            # assign them to new edge feature matrix
            new_edge[edge_indices] = edge_w

            # calculate attention logits for pooling of edges messege pass
            att_scores = self.att(edge_w)
            att_logits = F.softmax(att_scores, dim=0)
            # apply GRU update
            hidden_t = (att_logits.view(-1, 1) * edge_w).sum(dim=0).unsqueeze(dim=0)
            new_x.append(self.gru(x[node_idx].unsqueeze(dim=0), hidden_t).squeeze())
            
        return torch.stack(new_x), new_edge


class AGNN(Gnn):
    def __init__(self, in_channels, dim, out_size):
        super(AGNN, self).__init__()
        self.conv1 = AGNN_layer(in_channels, dim)
        self.conv2 = AGNN_layer(in_channels, dim)
        self.conv3 = AGNN_layer(in_channels, dim)
        self.linear = nn.Linear(in_channels, out_size)

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


class EAGNN(Gnn):
    def __init__(self, in_channels, dim, out_size):
        super(EAGNN, self).__init__()
        self.conv1 = EAGNN_layer(in_channels, dim)
        self.conv2 = EAGNN_layer(in_channels, dim)
        self.conv3 = EAGNN_layer(in_channels, dim)
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
        x, edge_w = self.conv1(x, edge_idx, edge_w)
        x, edge_w = self.conv2(x, edge_idx, edge_w)
        x, edge_w = self.conv3(x, edge_idx, edge_w)
        x = self.linear(x)
        
        device = x.get_device()
        batch = torch.zeros((N), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        return x




class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, global_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential (
            nn.Linear(2*node_in + edge_in + global_dim, edge_out),
            nn.ReLU(),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)
        
class EdgeModel_input(torch.nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, global_dim):
        super(EdgeModel_input, self).__init__()
        self.edge_mlp = nn.Sequential (
            nn.Linear(2*node_in + edge_in, edge_out),
            nn.ReLU(),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out
        

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, global_dim):
        super(NodeModel, self).__init__()
        self.node_mlp_1  = nn.Sequential (
            nn.Linear(node_in + edge_out, node_out),
            nn.ReLU(),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential (
            nn.Linear(global_dim + node_in + node_out, node_out),
            nn.ReLU(),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class NodeModel_input(torch.nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, global_dim):
        super(NodeModel_input, self).__init__()
        self.node_mlp_1  = nn.Sequential (
            nn.Linear(node_in + edge_out, node_out),
            nn.ReLU(),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential (
            nn.Linear(node_in + node_out, node_out),
            nn.ReLU(),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, global_dim):
        super(GlobalModel, self).__init__()
        self.global_mlp = nn.Sequential (
            nn.Linear(global_dim + node_out, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class ChanneledMetaLayer(nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None, num_channels=5):
        super(ChanneledMetaLayer, self).__init__()
        self.num_channels = num_channels
        self.layers = [MetaLayer(edge_model, node_model, global_model) for _ in range(num_channels)]

    def forward(self, *args):
        results = [layer(args) for layer in layers]
        [x_mat, edge_w_mat, global_vec_mat]   = [[results[i][j] for i in results] for j in results[0]]
        x = torch.stack(x_mat, dim=2)
        x = torch.mean(x, dim=2, keepdim=True)

        edge_w = torch.stack(edge_w_mat, dim=2)
        edge_w = torch.mean(edge_w, dim=2, keepdim=True)

        global_vec = torch.stack(global_vec_mat, dim=2)
        global_vec = torch.mean(global_vec, dim=2, keepdim=True)

        return x, edge_w, global_vec


class MetaGNN(torch.nn.Module):
    def __init__(self, node_in, edge_in, global_dim):
        super(MetaGNN, self).__init__()
        node_out = global_dim
        edge_out = global_dim
        self.global_dim = global_dim
        self.conv1 = MetaLayer(EdgeModel_input(node_in, edge_in, node_out, edge_out, global_dim), NodeModel_input(node_in, edge_in, node_out, edge_out, global_dim), GlobalModel(node_in, edge_in, node_out, edge_out, global_dim))
        self.conv2 = MetaLayer(EdgeModel(node_out, edge_out, node_out, edge_out, global_dim), NodeModel(node_out, edge_out, node_out, edge_out, global_dim), GlobalModel(node_out, edge_out, node_out, edge_out, global_dim))
        self.conv3 = MetaLayer(EdgeModel(node_out, edge_out, node_out, edge_out, global_dim), NodeModel(node_out, edge_out, node_out, edge_out, global_dim), GlobalModel(node_out, edge_out, node_out, edge_out, global_dim))
        self.linear = nn.Linear(global_dim, global_dim)
        
        def init_weights(m):
            if type(m) == nn.Linear:
        #        torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)
        
    def forward(self, sg):
        x, edge_idx, edge_w = sg
        x = x.float()
        E = edge_w
        device = x.get_device()
        N = len(x)
        batch = torch.zeros((N,), dtype=torch.long, device=device)
        global_vec = torch.normal(mean=0, std=0.01, size=(1,self.global_dim), device=device)
        x, edge_w, global_vec = self.conv1(x, edge_idx, edge_w, global_vec, batch)
        #print('X after: ', x)
        x, edge_w, global_vec = self.conv2(x, edge_idx, edge_w, global_vec, batch)
        x, edge_w, global_vec = self.conv3(x, edge_idx, edge_w, global_vec, batch)
        global_vec = self.linear(global_vec)
        
        return global_vec

class ChanneledMetaGNN(torch.nn.Module):
    def __init__(self, node_in, edge_in, global_dim):
        super(ChanneledMetaGNN, self).__init__()
        node_out = global_dim
        edge_out = global_dim
        self.global_dim = global_dim
        self.conv1 = ChanneledMetaLayer(EdgeModel_input(node_in, edge_in, node_out, edge_out, global_dim), NodeModel_input(node_in, edge_in, node_out, edge_out, global_dim), GlobalModel(node_in, edge_in, node_out, edge_out, global_dim))
        self.conv2 = ChanneledMetaLayer(EdgeModel(node_out, edge_out, node_out, edge_out, global_dim), NodeModel(node_out, edge_out, node_out, edge_out, global_dim), GlobalModel(node_out, edge_out, node_out, edge_out, global_dim))
        self.conv3 = ChanneledMetaLayer(EdgeModel(node_out, edge_out, node_out, edge_out, global_dim), NodeModel(node_out, edge_out, node_out, edge_out, global_dim), GlobalModel(node_out, edge_out, node_out, edge_out, global_dim))
        self.linear = nn.Linear(global_dim, global_dim)
        
        def init_weights(m):
            if type(m) == nn.Linear:
        #        torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)
        
    def forward(self, sg):
        x, edge_idx, edge_w = sg
        E = edge_w
        device = x.get_device()
        N = len(x)
        batch = torch.zeros((N,), dtype=torch.long, device=device)
        global_vec = torch.normal(mean=0, std=0.01, size=(1,self.global_dim), device=device)
        x, edge_w, global_vec = self.conv1(x, edge_idx, edge_w, global_vec, batch)
        x, edge_w, global_vec = self.conv2(x, edge_idx, edge_w, global_vec, batch)
        x, edge_w, global_vec = self.conv3(x, edge_idx, edge_w, global_vec, batch)
        global_vec = self.linear(global_vec)
        
        return global_vec




GNN_ARCHITECHTURE = {'SimpleGNN':SimpleGNN, 'GaussGNN':GaussGNN, 'AGNN':AGNN, 'EAGNN':EAGNN, 'MetaGNN':MetaGNN }
def build_gnn(cfg):
    gnn = GNN_ARCHITECHTURE[cfg.LISTENER.GNN](cfg.LISTENER.NODE_SIZE, cfg.LISTENER.EDGE_SIZE, cfg.LISTENER.GNN_OUTPUT)
    return gnn

if __name__ == '__main__':
    gnn = ChanneledMetaGNN(1, 2, 5)
    #gnn = SimpleGNN(1, 2, 4)
    x = torch.zeros((4, 1))
    y = torch.tensor([[0, 1, 1, 2],
                    [1, 0, 2, 3]], dtype=torch.long)
    w = torch.ones((4, 2))
    output = gnn((x,y,w))
    print(output)







