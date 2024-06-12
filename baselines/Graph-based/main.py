import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import DatasetLoader
import argparse
import json
import torch_geometric_temporal.nn.recurrent as R
import math
from models.evolvegcnh import EvolveGCNH

def temporal_signal_split(data_iterator, train_ratio: float = 0.8, eval_ratio=0.1):
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """

    train_snapshots = int(train_ratio * data_iterator.snapshot_count)
    eval_snapshots = int(eval_ratio * data_iterator.snapshot_count)
    
    train_iterator = data_iterator[0:train_snapshots]
    eval_iterator = data_iterator[train_snapshots:train_snapshots+eval_snapshots]
    test_iterator = data_iterator[train_snapshots+eval_snapshots:]

    return train_iterator, eval_iterator, test_iterator

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='r0')
parser.add_argument('--mode', type=str, default='rate')
parser.add_argument('--model_name', type=str, default='EvolveGCNH')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--window_size', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--pred_length', type=int, default=3)
parser.add_argument('--num_epochs', type=int, default=500)
args = parser.parse_args()

loader = DatasetLoader(data_name = args.data_name, mode=args.mode)
dataset = loader.get_dataset(lags = args.window_size)
train_dataset, eval_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.83, eval_ratio=0.04)
num_nodes = dataset.features[0].shape[0]
node_features = 1

class RNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, periods, pred_length):
        super(RNN, self).__init__()
        self.model_name = args.model_name
        if args.model_name == 'A3TGCN':
            self.recurrent = R.A3TGCN(node_features, hidden_dim, periods)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'DCRNN':
            self.recurrent = R.DCRNN(periods, hidden_dim, 1)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'DyGrEncoder':
            self.recurrent = R.DyGrEncoder(conv_out_channels=periods, conv_num_layers=1, conv_aggr="mean", lstm_out_channels=hidden_dim, lstm_num_layers=1)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'EvolveGCNH':
            self.recurrent = EvolveGCNH(num_nodes, periods)
            self.linear = torch.nn.Linear(periods, pred_length)
        elif args.model_name == 'EvolveGCNO':
            self.recurrent = R.EvolveGCNO(periods)
            self.linear = torch.nn.Linear(periods, pred_length)
        elif args.model_name == 'GCLSTM':
            self.recurrent = R.GCLSTM(periods, hidden_dim, 1)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'GConvGRU':
            self.recurrent = R.GConvGRU(periods, hidden_dim, 1)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'GConvLSTM':
            self.recurrent = R.GConvLSTM(periods, hidden_dim, 1)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'LRGCN':
            self.recurrent = R.LRGCN(periods, hidden_dim, 1, 1)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        elif args.model_name == 'MPNNLSTM':
            self.recurrent = R.MPNNLSTM(periods, hidden_dim, num_nodes, 1, 0.5)
            self.linear = torch.nn.Linear(2 * hidden_dim + periods, pred_length)
        elif args.model_name == 'TGCN':
            self.recurrent = R.TGCN(periods, hidden_dim)
            self.linear = torch.nn.Linear(hidden_dim, pred_length)
        else:
            assert False

    def forward(self, x, edge_index, edge_weight, h=None, c=None):
        h_0, c_0 = None, None
        if self.model_name in ['A3TGCN']:
            h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
        elif self.model_name in ['TGCN']:
            h_0 = self.recurrent(x, edge_index, edge_weight, h)
            h = F.relu(h_0)
            h = self.linear(h)
        elif self.model_name in ['EvolveGCNH', 'EvolveGCNO', 'GConvGRU', 'MPNNLSTM', 'DCRNN']:
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
        elif self.model_name in ['GCLSTM', 'GConvLSTM', 'LRGCN']:
            h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
            h = F.relu(h_0)
            h = self.linear(h)
        elif self.model_name in ['DyGrEncoder']:
            h, h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
            h = F.relu(h)
            h = self.linear(h)
        return h, h_0, c_0
        

for seed in range(2):
    args.seed = seed
    if not os.path.exists(f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/metrics.json'):
        if not os.path.exists(f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}'):
            os.makedirs(f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}')
        model = RNN(node_features = 1, hidden_dim=args.hidden_dim, periods = args.window_size, pred_length=args.pred_length)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        best_cost = 1e20
        for epoch in tqdm(range(args.num_epochs)):
            model.train()
            cost = 0
            h, c = None, None
            e = torch.empty(num_nodes, node_features)
            for time, snapshot in enumerate(train_dataset):
                if args.model_name in ['A3TGCN', 'EvolveGCNH', 'EvolveGCNO', 'GConvGRU', 'MPNNLSTM', 'DCRNN']:
                    snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y = snapshot.x.to(args.device), snapshot.edge_index.to(args.device), snapshot.edge_attr.to(args.device), snapshot.y.to(args.device)
                elif args.model_name in ['TGCN']:
                    snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y = snapshot.x.to(args.device), snapshot.edge_index.to(args.device), snapshot.edge_attr.to(args.device), snapshot.y.to(args.device)
                elif args.model_name in ['GCLSTM', 'GConvLSTM', 'LRGCN', 'DyGrEncoder']:
                    snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y = snapshot.x.to(args.device), snapshot.edge_index.to(args.device), snapshot.edge_attr.to(args.device), snapshot.y.to(args.device)
                y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
                cost = cost + torch.mean((y_hat-snapshot.y)**2)
            cost = cost / (time+1)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                model.eval()
                cost = 0
                h, c = None, None
                preds = []
                golds = []
                for time, snapshot in enumerate(test_dataset):
                    y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
                    cost = cost + torch.mean((y_hat-snapshot.y)**2)
                    preds.append(y_hat)
                    golds.append(snapshot.y)
                cost = cost / (time+1)
                cost = cost.item()
                if cost < best_cost:
                    best_cost = cost
                    print(math.sqrt(cost))
                    torch.save(model, f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/model.pt')
                    for time in range(len(preds)):
                        torch.save(preds[time], f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/pred_{time}.pt')
                        torch.save(golds[time], f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/gold_{time}.pt')

    model = torch.load(f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/model.pt')
    model.eval()
    with torch.no_grad():
        cost = 0
        mae = 0
        h, c = None, None
        for time, snapshot in enumerate(test_dataset):
            y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
            torch.save(y_hat, f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/pred_{time}.pt')
            torch.save(snapshot.y, f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/gold_{time}.pt')
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
            mae = mae + torch.mean(torch.abs(y_hat-snapshot.y))

        cost = (cost / (time+1)).item()
        mae = (mae / (time+1)).item()
        ans = {
            "RMSE": math.sqrt(cost),
            "MAE": mae,
        }
        print(ans)
        json.dump(ans, open(f'results/{args.mode}/{args.data_name}/{args.model_name}/{args.seed}/metrics.json', 'w'))