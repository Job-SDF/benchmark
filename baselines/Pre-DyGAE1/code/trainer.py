import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.dataloading import GraphDataLoader
from graphSampler import SubgraphIterator
from utils import calc_mrr, get_subset_g, time_metric
import os
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, g, data, model, device, lr, num_epochs, sample_size, checkpoint_path, results_path, scores_path, eval_step, k=10, optimizer=None, node_embedding_path=None, time_embedding_path=None, old_g=None, time="no", load_time_embedding_path=None, **kwargs):
        self.g = g

        self.model = model
        self.data = data

        num_nodes = g.num_nodes()
        num_rels = data.num_rels

        
        

        train_g = get_subset_g(g, g.edata["train_mask"], num_rels)

        eval_g = get_subset_g(g, g.edata["eval_mask"], num_rels)
        eval_g.edata["norm"] = dgl.norm_by_dst(eval_g).unsqueeze(-1)

        test_g = get_subset_g(g, g.edata["test_mask"], num_rels)
        test_g.edata["norm"] = dgl.norm_by_dst(test_g).unsqueeze(-1)

        src, dst = g.edges()
        triplets = torch.stack([src, g.edata["etype"], dst], dim=1)

        self.pos_num_nodes = model.pos_num_nodes
        self.skill_num_nodes = model.skill_num_nodes

        self.train_g = train_g
        self.eval_g = eval_g
        self.test_g = test_g
        self.triplets = triplets
        self.eval_nids = torch.arange(0, num_nodes)
        self.test_nids = torch.arange(0, num_nodes)
        self.train_mask = g.edata['train_mask']
        self.eval_mask = g.edata['eval_mask']
        self.test_mask = g.edata['test_mask']

        self.checkpoint_path = checkpoint_path
        self.results_path = results_path
        self.scores_path = scores_path
        self.node_embedding_path = node_embedding_path
        self.time_embedding_path = time_embedding_path
        self.num_rels = num_rels
        self.device = device
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.sample_size = sample_size
        self.node_ids = g.ndata['nodes']
        self.eval_step = eval_step
        self.k = k
        self.skill2cluster = self.g.ndata['ncluster']
        self.num_nodes = num_nodes
        self.time = time

        self.load_time_embedding_path = load_time_embedding_path

    def get_train_dataloader(self, g, k=10):
        subg_iter = SubgraphIterator(g, self.num_rels, sample_size=self.sample_size,
                                     num_epochs=1, k=k, time=self.time)  
        dataloader = GraphDataLoader(
            subg_iter, batch_size=1, collate_fn=lambda x: x[0], shuffle=True)
        return dataloader

    def get_eval_dataloader(self, g, k=10):
        subg_iter = SubgraphIterator(g, self.num_rels, sample_size=self.sample_size,
                                     num_epochs=1, k=10, time=self.time)  
        dataloader = GraphDataLoader(
            subg_iter, batch_size=1, collate_fn=lambda x: x[0])
        return dataloader

    def train(self):
        train_dataloader = self.get_train_dataloader(self.train_g, self.k)
        model = self.model.to(self.device)
        device = self.device
        optimizer = self.optimizer
        best_mae = 0
        best_loss = 100.0
        step = 0
        with tqdm(range(self.num_epochs), desc="Train") as bar:
            for epoch in bar:
                for batch_data in train_dataloader:  

                    

                    step += 1
                    model.train()
                    if 'diff_labels' in self.g.edata:
                        g, train_nids, edges, labels, edge_labels, diff_labels = batch_data
                        g, train_nids, edges, labels, edge_labels, diff_labels = g.to(device), train_nids.to(
                            device), edges.to(device), labels.to(device), edge_labels.to(device), diff_labels.to(device)
                    else:
                        g, train_nids, edges, labels, edge_labels = batch_data
                        g, train_nids, edges, labels, edge_labels = g.to(device), train_nids.to(
                            device), edges.to(device), labels.to(device), edge_labels.to(device)
                        diff_labels = None

                    outputs = model(g, train_nids, edges, labels, edge_labels, self.skill2cluster.to(
                        device), diff_labels=diff_labels)

                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)  
                    optimizer.step()
                    bar.set_description(
                        f"mae:{outputs.real_mae}, loss:{outputs.loss}, lp_loss: {outputs.lp_loss}, rg_loss:{outputs.rg_loss}, rg_loss_pos:{outputs.rg_loss_pos}, rg_loss_neg:{outputs.rg_loss_neg}, rank_loss: {outputs.rank_loss}, con_loss: {outputs.con_loss}, kl_loss: {outputs.kl_loss}, diff_loss: {outputs.diff_loss}")
                        
                    if step % self.eval_step == 0:
                        logger.info(f"epoch {epoch},step {step}")
                        
                        logger.info(
                            f"Trainloss:{outputs.loss}, lp_loss: {outputs.lp_loss}, rg_loss:{outputs.rg_loss}, rg_loss_pos:{outputs.rg_loss_pos}, rg_loss_neg:{outputs.rg_loss_neg}, rank_loss: {outputs.rank_loss}, con_loss: {outputs.con_loss}, kl_loss: {outputs.kl_loss}, diff_loss: {outputs.diff_loss}")
                        metric = self.evaluate()
                        

                        if best_mae < metric['MRR'] or (best_mae <= metric['MRR'] and best_loss > outputs.loss):
                            best_mae = metric['MRR']
                            best_loss = outputs.loss

                            if self.time == 'yes':
                                if self.time_embedding_path:
                                    torch.save(
                                        (self.real_mu_time_embeddings, self.real_sigma_time_embeddings), self.time_embedding_path)
                                if self.node_embedding_path:
                                    torch.save(
                                        (self.real_mu_embeddings, self.real_sigma_embeddings), self.node_embedding_path)
                            else:
                                self.save_model(step)
                                if self.node_embedding_path:
                                    torch.save(
                                        (self.real_mu_embeddings, self.real_sigma_embeddings), self.node_embedding_path)

    def gain_time_embedding(self):
        model = self.model
        device = self.device
        model.eval()
        gaussian = model.gaussian

        train_dataloader = self.get_train_dataloader(self.train_g, self.k)
        with torch.no_grad():
            for batch_data in train_dataloader:  
                g, train_nids, edges, labels, edge_labels = batch_data

            g, train_nids, edges, labels, edge_labels = g.to(device), train_nids.to(
                device), edges.to(device), labels.to(device), edge_labels.to(device)

            _, (mu_embdding, sigma_embdding), (mu_time_embdding,
                                               sigma_time_embdding) = model(g, train_nids, edges, std='no')
        return (mu_embdding, sigma_embdding), (mu_time_embdding, sigma_time_embdding)

    def evaluate(self, mode='eval'):
        model = self.model
        device = self.device
        model.eval()
        gaussian = model.gaussian
        train_dataloader = self.get_train_dataloader(self.train_g, self.k)
        metric = {}
        with torch.no_grad():
            for batch_data in train_dataloader:  
                if 'diff_labels' in self.g.edata:
                    g, train_nids, edges, labels, edge_labels, diff_labels = batch_data
                    g, train_nids, edges, labels, edge_labels, diff_labels = g.to(device), train_nids.to(
                        device), edges.to(device), labels.to(device), edge_labels.to(device), diff_labels.to(device)
                else:
                    g, train_nids, edges, labels, edge_labels = batch_data
                    g, train_nids, edges, labels, edge_labels = g.to(device), train_nids.to(
                        device), edges.to(device), labels.to(device), edge_labels.to(device)

                train_embed, (mu_embedding, sigma_embedding), (mu_time_embdding,
                                                               sigma_time_embdding), (mu, sigma) = model.encoder(g, train_nids, std='no')

                if self.node_embedding_path:
                    real_mu_embeddings = torch.zeros(
                        self.pos_num_nodes + self.skill_num_nodes, mu_embedding.shape[-1]).to(self.device)
                    real_mu_embeddings[train_nids] = mu_embedding
                    real_sigma_embeddings = torch.zeros(
                        self.pos_num_nodes + self.skill_num_nodes, mu_embedding.shape[-1]).to(self.device)
                    real_sigma_embeddings[train_nids] = sigma_embedding
                    self.real_mu_embeddings = real_mu_embeddings
                    self.real_sigma_embeddings = real_sigma_embeddings

                if self.time_embedding_path:
                    real_mu_time_embeddings = torch.zeros(
                        self.pos_num_nodes + self.skill_num_nodes, mu_embedding.shape[-1]).to(self.device)
                    real_mu_time_embeddings[train_nids] = mu_time_embdding
                    real_sigma_time_embeddings = torch.zeros(
                        self.pos_num_nodes + self.skill_num_nodes, mu_embedding.shape[-1]).to(self.device)
                    real_sigma_time_embeddings[train_nids] = sigma_time_embdding
                    self.real_mu_time_embeddings = real_mu_time_embeddings
                    self.real_sigma_time_embeddings = real_sigma_time_embeddings

                import copy
                if self.time == 'yes':
                    if model.load_node_embed:
                        embed = copy.deepcopy(model.mu_embedding.weight)
                    else:
                        embed = torch.randn(model.num_nodes, model.h_dim).to(train_embed.device)
                    embed[train_nids] = train_embed
                else:
                    if train_embed.shape[0] != model.num_nodes:
                        embed = torch.randn(model.num_nodes, model.h_dim).to(train_embed.device)
                        embed[train_nids] = train_embed
                    else:
                        embed = train_embed
                all_triplets_pos = torch.range(
                    0, self.pos_num_nodes - 1).repeat_interleave(self.skill_num_nodes).unsqueeze(0)
                all_triplets_skill = torch.range(
                    self.pos_num_nodes, self.skill_num_nodes + self.pos_num_nodes - 1).repeat(self.pos_num_nodes).unsqueeze(0)
                all_triplets_relation = torch.zeros(
                    all_triplets_skill.shape[1]).unsqueeze(0)
                all_triplets = torch.cat(
                    [all_triplets_pos, all_triplets_relation, all_triplets_skill], dim=0).T.long().to(embed.device)

                all_lp_score = model.calc_lp_score(embed, all_triplets)
                all_rg_score = model.calc_rg_score(embed, all_triplets)
                regression_matrix_score = self.data.scaler.inverse_transform(all_rg_score.detach().cpu().numpy())

                all_rg_matrix = torch.zeros(
                    self.pos_num_nodes, self.skill_num_nodes).to(all_rg_score.device)
                all_rg_matrix[all_triplets[:, 0], all_triplets[:,
                                                               2] - self.pos_num_nodes] = all_rg_score

                all_lp_matrix = torch.zeros(
                    self.pos_num_nodes, self.skill_num_nodes).to(all_lp_score.device)
                all_lp_matrix[all_triplets[:, 0], all_triplets[:,
                                                               2] - self.pos_num_nodes] = all_lp_score

                all_labels = torch.zeros(
                    self.pos_num_nodes, self.skill_num_nodes).to(all_rg_score.device)
                all_labels[self.triplets[self.eval_mask][:, 0], self.triplets[self.eval_mask]
                           [:, 2] - self.pos_num_nodes] = self.eval_g.edata['edge_labels'].to(device)

                if self.time == 'no':
                    all_labels = torch.zeros(
                        self.pos_num_nodes, self.skill_num_nodes).to(all_lp_score.device)
                    all_labels[self.triplets[self.eval_mask][:, 0], self.triplets[self.eval_mask][:, 2] -
                               self.pos_num_nodes] = self.eval_g.edata['edge_labels'].to(all_lp_score.device)

                    all_lp_matrix = torch.zeros(
                        self.pos_num_nodes, self.skill_num_nodes).to(all_rg_score.device)
                    all_lp_matrix[all_triplets[:, 0], all_triplets[:, 2] -
                                  self.pos_num_nodes] = torch.nn.Sigmoid()(all_lp_score)
                    all_lp_matrix[self.triplets[self.train_mask | self.test_mask][:, 0],
                                  self.triplets[self.train_mask | self.test_mask][:, 2] - self.pos_num_nodes] = 0

                    all_rg_matrix = torch.zeros(
                        self.pos_num_nodes, self.skill_num_nodes).to(all_rg_score.device)
                    all_rg_matrix[all_triplets[:, 0], all_triplets[:,
                                                                   2] - self.pos_num_nodes] = all_rg_score
                    all_rg_matrix[self.triplets[self.train_mask | self.test_mask][:, 0],
                                  self.triplets[self.train_mask | self.test_mask][:, 2] - self.pos_num_nodes] = 0

                metric['regression'] = time_metric(
                    all_labels, all_rg_matrix)
                metric['link_prediction'] = time_metric(
                    all_labels, all_lp_matrix)
                metric['mrr'] = max(
                    metric['regression']['MRR'], metric['link_prediction']['MRR'])
                final_metric = {m: metric['link_prediction'][m]
                                for m in ['AUC', 'Hits@1', 'Hits@3', "MRR"]}
                final_metric.update({m: metric['regression'][m] for m in [
                                    'EGM', 'MAE', 'MAPE', "RMSE"]})
                logger.info(metric)
                logger.info(final_metric)

                
                if mode == 'test':
                    np.save(self.scores_path[:-3] + '.npy', regression_matrix_score)
                    torch.save(all_rg_matrix, self.scores_path)
                    torch.save(all_lp_matrix, self.results_path)

        return final_metric

    def save_model(self, step):
        torch.save(self.model.state_dict(), self.checkpoint_path)
