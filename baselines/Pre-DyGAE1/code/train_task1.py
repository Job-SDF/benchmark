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


if __name__ == "__main__":
    data = JDDataset(reverse=False, name=args.data_name, raw_dir=f'data/{args.data_name}', train_path=args.train_path, eval_path=args.eval_path, test_path=args.test_path)
    g = data[0]
    num_nodes = g.num_nodes()
    num_rels = data.num_rels

    entities = pd.read_csv(f'data/{args.data_name}/entities.dict', sep='\t', header=None)
    pos_num_nodes = (entities[1]<30000).sum()
    skill_num_nodes = (entities[1]>=30000).sum()

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

    model = LinkPredict(num_nodes, pos_num_nodes, skill_num_nodes, num_rels, cross_attn=args.cross_attn, time=args.time, embedding=entity2embedding,
                        rg_weight=args.rg_weight, lp_weight=args.lp_weight, rank_weight=args.rank_weight, con_weight=args.con_weight,
                        gaussian=args.gaussian, bias=args.bias, initial_embedding=args.initial_embedding,
                        rg_loss_fn=rg_loss_fn[args.rg_loss_fn], rg_activate_fn=rg_activate_fn[args.rg_activate_fn], scaler=data.scaler).to(args.device)

    params = []
    if args.fix_emb == 'yes':
        for param in model.encoder.emb.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if 'emb' not in name:
                params.append({"params": param})
            else:
                print(f"fix the {name}")
    else:
        for name, param in model.named_parameters():
            params.append({"params": param})

    opt = torch.optim.Adam(params, lr=args.lr)

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
        "optimizer": opt,
        "results_path": args.results_path,
        "scores_path": args.scores_path,
        "node_embedding_path": args.node_embedding_path,
        "k": args.k,
    }
    trainer = Trainer(**config)
    trainer.train()
