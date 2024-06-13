import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphormerLayer
import torch.nn.functional as F

class Graphormer(nn.Module):
    def __init__(self, feat_size, hidden_size, num_heads):
        super().__init__()
        
        self.conv1 = GraphormerLayer(
            feat_size=feat_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.conv2 = GraphormerLayer(
            feat_size=feat_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, x, bias):
        x, bias = x.unsqueeze(0), bias.unsqueeze(0)
        h = F.relu(self.conv1(x, bias))
        h = self.dropout(h)
        h = self.conv2(h, bias).squeeze(0)
        return self.dropout(h)