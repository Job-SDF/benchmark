import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphormerLayer
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from models.rgcn import RGCN


class TemporalEmb(nn.Module):
    def __init__(self, num_nodes, h_dim, e_dim=10, real_id_nodes=None, out_channels=2, num_rels=1, adaptive='yes'):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)

        self.sigma_linear = nn.Linear(h_dim, h_dim)
        self.mu_linear = nn.Linear(h_dim, h_dim)

        self.sigma_linear = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU()) 
        self.mu_linear = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU()) 

        self.mu_emb = nn.Embedding(num_nodes, h_dim)
        self.sigma_emb = nn.Embedding(num_nodes, h_dim)
        self.inital_emb_weight = False

        self.rgcn = RGCN(h_dim, num_rels * 2)
        if real_id_nodes is None:
            real_num_nodes = num_nodes
        else:
            real_num_nodes = real_id_nodes.shape[0]
        
        self.real_id_nodes = real_id_nodes

        self.AGCRN = AGCRN(number_of_nodes = real_num_nodes, in_channels = h_dim, out_channels = out_channels, K = 2, embedding_dimensions = e_dim)
        
        self.ensemble = nn.Linear(h_dim + out_channels, h_dim)

        self.mu_embedding = nn.Linear(h_dim, h_dim)
        self.sigma_embedding = nn.Linear(h_dim, h_dim)

        self.adaptive = adaptive
    
    def load_emb_weight(self, mu_emb_weight, sigma_emb_weight):
        self.sigma_emb.weight = nn.Parameter(sigma_emb_weight) 
        self.mu_emb.weight = nn.Parameter(mu_emb_weight) 
        self.inital_emb_weight = True
    
    def load_emb_graph(self, graph_inital_emb):
        if graph_inital_emb == None:
            self.e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.real_id_nodes.shape[0], e_dim), gain=nn.init.calculate_gain('relu')))
        else:
            self.e = nn.Parameter(graph_inital_emb[self.real_id_nodes])


    def forward(self, x, g):
        if self.inital_emb_weight:
            mu = self.mu_emb(x)
            sigma = self.sigma_emb(x)
        else:
            embedding = self.emb(x)
            new_embedding = self.rgcn(g, embedding)
            if self.adaptive == 'yes':
                ada_embedding = self.AGCRN(embedding.unsqueeze(0), self.e).squeeze(0)
                ensemble_embedding = torch.cat([new_embedding, ada_embedding], dim=-1)
                ensemble_embedding = self.ensemble(ensemble_embedding)
            else:
                ensemble_embedding = new_embedding
            mu = self.mu_embedding(ensemble_embedding)
            sigma = self.sigma_embedding(ensemble_embedding)
        return mu, sigma