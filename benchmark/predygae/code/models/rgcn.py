import torch
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
import torch.nn.functional as F
import dgl

class RGCN(nn.Module):
    def __init__(self, h_dim, num_rels):
        super().__init__()
        
        self.conv1 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="bdd",
            num_bases=128,
            self_loop=True,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="bdd",
            num_bases=128,
            self_loop=True,
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, x):
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"]))
        h = self.dropout(h)
        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
        return self.dropout(h)

