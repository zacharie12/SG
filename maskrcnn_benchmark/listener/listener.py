import torch
import torch.nn as nn
import typing 
from .cnn import *
from .gnn import *
from .listener_end import *
import torch.nn.functional as F

class Listener(nn.Module):
    def __init__(self, gnn_t : Gnn, cnn_t : Cnn, listener_end_t : ListenerEnd):
        super(Listener, self).__init__()
        self.gnn = gnn_t
        self.cnn = cnn_t
        self.listener_end = listener_end_t
        
    def forward(self, sg, image):
        # Apply softmax to nodes and edges
        (node_w, edge_index, edge_w) = sg
        edge_w = F.softmax(edge_w, dim=1)
        node_w = F.softmax(node_W, dim=1)
        sg = (node_w, edge_index, edge_w)
        g_feature = self.gnn(sg)
        im_feature = self.cnn(image)
        batch_size = im_feature.size(0)
        g_feature = g_feature.repeat(batch_size, 1)
        g_feature = g_feature.reshape(batch_size, -1)

        feature = torch.cat((g_feature, im_feature), dim=1)
        #return F.softmax(self.listener_end(feature), dim=0)
        return self.listener_end(feature)

def build_listener(cfg):
    listener = Listener(build_gnn(cfg), build_cnn(cfg), build_end(cfg))
    return listener
    
if __name__ == '__main__':
    gnn = SimpleGCN(1, 2)
    cnn = GroupCNN(24, 24, 3, 4)
    fc = FC(5)
    listener = Listener(gnn, cnn, fc)

    x = torch.zeros((4, 1))
    im = torch.zeros((3, 3, 24, 24)) 
    y = torch.tensor([[0, 1, 1, 2],
                    [1, 0, 2, 3]], dtype=torch.long)
    w = torch.ones((4, 2))
    sg = (x, y, w)

    output = listener(sg, im)
    print(output)        