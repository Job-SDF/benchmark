import dgl
import numpy as np
import torch
import torch.nn as nn
import tqdm
from dgl.nn.pytorch import RelGraphConv
from mydataset import JDDataset
from args import args
from utils import get_subset_g
from model import LinkPredict
from trainer import Trainer
from utils import calc_mrr
import logging
import torch.nn.functional as F
from torchmetrics.regression import TweedieDevianceScore
import pandas as pd
from torch_geometric_temporal.nn.recurrent import AGCRN

if __name__ == "__main__":
    data = JDDataset(reverse=False, name=args.data_name, raw_dir=f'data/{args.data_name}', train_path=args.train_path, eval_path=args.eval_path, test_path=args.test_path)
    g = data[0]
    num_nodes = g.num_nodes()
    num_rels = data.num_rels
    entities = pd.read_csv(f'data/{args.data_name}/entities.dict', sep='\t', header=None)
    pos_num_nodes = (entities[1]<30000).sum()
    skill_num_nodes = (entities[1]>=30000).sum()
    real_id_nodes = np.unique((g.edges()[0].numpy(), g.edges()[1].numpy()))

    if args.bias == 'yes':
        entity2embedding = torch.load(
            f'data/{args.data_name}/entity2embedding.pt')
    else:
        entity2embedding = None

    rg_loss_fn = {
        "l1": F.l1_loss,
        "mse": F.mse_loss,
        "tweedie": TweedieDevianceScore(1.5)
    }

    rg_activate_fn = {
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "leakyrelu": nn.LeakyReLU()
    }

    if args.graph_inital_emb_path:
        graph_inital_emb = torch.load(args.graph_inital_emb_path)

    model = LinkPredict(num_nodes, pos_num_nodes, skill_num_nodes, num_rels, cross_attn=args.cross_attn, time=args.time, embedding=entity2embedding,
                        rg_weight=args.rg_weight, lp_weight=args.lp_weight, rank_weight=args.rank_weight, con_weight=args.con_weight, diff_weight=args.diff_weight,
                        gaussian=args.gaussian, bias=args.bias, initial_embedding=args.initial_embedding, e_dim=args.e_dim,
                        rg_loss_fn=rg_loss_fn[args.rg_loss_fn], rg_activate_fn=rg_activate_fn[args.rg_activate_fn], real_id_nodes=real_id_nodes, scaler=data.scaler).to(args.device)

    if args.load_state_path:
        model.load_state_dict(torch.load(args.load_state_path), strict=False)
    model.load_node_embedding(*torch.load(args.load_node_embedding_path))
    model.temporal_emb.load_emb_graph(graph_inital_emb)
    model.load_node_temporal_embedding(*torch.load(args.load_time_embedding_path))

    config = {
        "g": g,
        "data": data,
        "model": model,
        "device": args.device,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "eval_step": args.eval_step,
        "sample_size": args.sample_size,
        "checkpoint_path": args.checkpoint_path,
        "log_path": args.log_path,
        "results_path": args.results_path,
        "scores_path": args.scores_path,
        "k": args.k,
        "time_embedding_path": args.time_embedding_path,
        "node_embedding_path": args.node_embedding_path,
        "time": "yes"
    }
    trainer = Trainer(**config)
    trainer.evaluate(mode='test')
