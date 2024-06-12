from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN, A3TGCN, DCRNN, DyGrEncoder, EvolveGCNH, EvolveGCNO, GCLSTM, GConvGRU, GConvLSTM, LRGCN, MPNNLSTM, TGCN

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, TwitterTennisDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn as nn
from dataset import DatasetLoader
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='r1')
parser.add_argument('--mode', type=str, default='count')
parser.add_argument('--model_name', type=str, default='AGCRN')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--window_size', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--pred_length', type=int, default=3)
args = parser.parse_args()

loader = DatasetLoader(data_name = args.data_name, mode=args.mode)
dataset = loader.get_dataset(lags = args.window_size)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)

num_nodes = train_dataset.features[0].shape[0]
epochs = 200
node_features = 16

class RecurrentGCN(torch.nn.Module):
    def __init__(self, number_of_nodes, node_features, model_name = 'DyGrEncoder'):
        super(RecurrentGCN, self).__init__()
        self.emb = nn.Embedding(number_of_nodes, node_features)
        self.model_name = model_name
        model_mp = {
            "A3TGCN": A3TGCN(1, 32, 4),
            'AGCRN': AGCRN(number_of_nodes = number_of_nodes,in_channels = node_features,out_channels = 2,K = 2,embedding_dimensions = 4),
            "DCRNN": DCRNN(node_features, 32, 1),
            "DyGrEncoder": DyGrEncoder(conv_out_channels=16, conv_num_layers=1, conv_aggr="mean", lstm_out_channels=32, lstm_num_layers=1),
            "EvolveGCNH": EvolveGCNH(number_of_nodes, node_features),
            "EvolveGCNO": EvolveGCNO(node_features),
            "GCLSTM": GCLSTM(node_features, 32, 1),
            "GConvGRU": GConvGRU(node_features, 32, 1),
            "GConvLSTM": GConvLSTM(node_features, 32, 1),
            # "LRGCN": LRGCN(node_features, 32, 1, 1),
            "MPNNLSTM": MPNNLSTM(node_features, 32, number_of_nodes, 1, 0.5),
            "TGCN": TGCN(node_features, 32),
        }
        self.recurrent = model_mp[model_name]

    def forward(self, x, e, h, h_0=None, c_0=None):
        x = self.emb(x)
        if self.model_name in ['A3TGCN', 'DCRNN', 'EvolveGCNH', 'EvolveGCNO', 'GConvGRU', 'MPNNLSTM']:
            if self.model_name in ['A3TGCN']:
                x = x.view(x.shape[0], 1, x.shape[1])
            h = self.recurrent(x, e, h)
            y = F.relu(h)
            user_emb = y[:loader._dataset['user_node_num']]
            item_emb = y[loader._dataset['user_node_num']:]
            y = user_emb@item_emb.T
            return y
        elif self.model_name in ['AGCRN']:
            h = self.recurrent(x, e, h)
            y = F.relu(h)
            user_emb = y[0,:loader._dataset['user_node_num']]
            item_emb = y[0,loader._dataset['user_node_num']:]
            y = user_emb@item_emb.T
            return y, h
        elif self.model_name in ['TGCN']:
            h = self.recurrent(x, e, h, h_0)
            y = F.relu(h)
            user_emb = y[:loader._dataset['user_node_num']]
            item_emb = y[loader._dataset['user_node_num']:]
            y = user_emb@item_emb.T
            return y, h
        elif self.model_name in ['GCLSTM', 'GConvLSTM', 'LRGCN']:
            h_0, c_0 = self.recurrent(x, e, h, h_0, c_0)
            y = F.relu(h_0)
            user_emb = y[:loader._dataset['user_node_num']]
            item_emb = y[loader._dataset['user_node_num']:]
            y = user_emb@item_emb.T
            return y, h_0, c_0
        elif self.model_name in ['DyGrEncoder']:
            h, h_0, c_0 = self.recurrent(x, e, h, h_0, c_0)
            y = F.relu(h)
            user_emb = y[:loader._dataset['user_node_num']]
            item_emb = y[loader._dataset['user_node_num']:]
            y = user_emb@item_emb.T
            return y, h_0, c_0
        
        
model = RecurrentGCN(number_of_nodes=num_nodes, node_features = node_features, model_name=args.model_name).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
e = torch.empty(num_nodes, 4).to(args.device)
torch.nn.init.xavier_uniform_(e)

for epoch in tqdm(range(epochs)):
    cost = 0
    h, c = None, None
    for time, snapshot in enumerate(train_dataset):
        snapshot = snapshot.to(args.device)
        if args.model_name in ['A3TGCN', 'DCRNN', 'EvolveGCNH', 'EvolveGCNO', 'GConvGRU', 'MPNNLSTM']:
            x = snapshot.x.view(num_nodes).long()
            y_hat = model(x, snapshot.edge_index, snapshot.edge_attr)
        elif args.model_name in ['TGCN']:
            x = snapshot.x.view(num_nodes).long()
            y_hat, h = model(x, snapshot.edge_index, snapshot.edge_attr, h)
        elif args.model_name in ['AGCRN']:
            x = snapshot.x.view(1, num_nodes).long()
            y_hat, h = model(x, e, h)
        elif args.model_name in ['DyGrEncoder', 'GCLSTM', 'GConvLSTM', 'LRGCN']:
            x = snapshot.x.view(num_nodes).long()
            y_hat, h, c = model(x, snapshot.edge_index, snapshot.edge_attr, h, c)
        y = torch.tensor(snapshot.y).to(x.device)
        cost = cost + torch.mean((y_hat-y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
for time, snapshot in enumerate(test_dataset):
    snapshot = snapshot.to(args.device)
    if args.model_name in ['A3TGCN', 'DCRNN', 'EvolveGCNH', 'EvolveGCNO', 'GConvGRU', 'MPNNLSTM']:
        x = snapshot.x.view(num_nodes).long()
        y_hat = model(x, snapshot.edge_index, snapshot.edge_attr)
    elif args.model_name in ['TGCN']:
        x = snapshot.x.view(num_nodes).long()
        y_hat, h = model(x, snapshot.edge_index, snapshot.edge_attr, h)
    elif args.model_name in ['AGCRN']:
        x = snapshot.x.view(1, num_nodes).long()
        y_hat, h = model(x, e, h)
    elif args.model_name in ['DyGrEncoder', 'GCLSTM', 'GConvLSTM', 'LRGCN']:
        x = snapshot.x.view(num_nodes).long()
        y_hat, h, c = model(x, snapshot.edge_index, snapshot.edge_attr, h, c)
    y = torch.tensor(snapshot.y).to(y_hat.device)
    m = {}
    m[args.model_name] = metric(y, y_hat)
    import pandas as pd
    pd.DataFrame(m).T.to_csv(f'results/{args.data_name}/{args.model_name}_{args.seed}.tsv')