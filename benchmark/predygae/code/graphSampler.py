import sys
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import RelGraphConv
from mydataset import JDDataset
import json

class GlobalUniform:
    def __init__(self, g, sample_size):
        self.sample_size = sample_size
        self.eids = np.arange(g.num_edges())

    def sample(self):
        
        return torch.from_numpy(self.eids)


class NegativeSampler:
    def __init__(self, k=10):  
        self.k = k

    def sample(self, pos_samples, num_nodes):
        batch_size = len(pos_samples)
        neg_batch_size = batch_size * self.k
        neg_samples = np.tile(pos_samples, (self.k, 1))

        values_0 = np.random.randint(pos_samples[:,0].max(), size=neg_batch_size)
        values_2 = np.random.randint(pos_samples[:,0].max() + 1, pos_samples[:,2].max(), size=neg_batch_size)
        choices = np.random.uniform(size=neg_batch_size)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values_0[subj]
        neg_samples[obj, 2] = values_2[obj]
        samples = np.concatenate((pos_samples, neg_samples))
        

        
        labels = np.zeros(batch_size * (self.k + 1), dtype=np.float32)
        labels[:batch_size] = 1

        return torch.from_numpy(samples), torch.from_numpy(labels)

class Time_NegativeSampler:
    def __init__(self, k=10):  
        self.k = k

    def sample(self, pos_samples, num_nodes):
        batch_size = len(pos_samples)

        pos_num_nodes = pos_samples[:,0].max() + 1
        skill_num_nodes = pos_samples[:,2].max() - pos_samples[:,2].min() + 1
        ori_matrix = torch.zeros(pos_num_nodes, skill_num_nodes)
        ori_matrix[pos_samples[:,0], pos_samples[:,2] - pos_samples[:,2].min()] = 1
        neg_matrix_index = (ori_matrix == 0).nonzero()

        neg_samples = np.zeros([neg_matrix_index.shape[0], 3])
        neg_samples[:,0] = neg_matrix_index[:,0]
        neg_samples[:,2] = neg_matrix_index[:,1] + pos_num_nodes
        neg_samples = neg_samples.astype(np.long)

        neg_batch_size = min(batch_size * self.k, neg_samples.shape[0])

        neg_samples = neg_samples[np.random.choice(neg_samples.shape[0], neg_batch_size, replace=False)]
        samples = np.concatenate((pos_samples, neg_samples))

        
        labels = np.zeros(batch_size + neg_batch_size, dtype=np.float32)
        labels[:batch_size] = 1

        return torch.from_numpy(samples), torch.from_numpy(labels)


class SubgraphIterator:
    def __init__(self, g, num_rels, sample_size=3000, num_epochs=1, k=10, time="no"):
        self.g = g
        self.num_rels = num_rels
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.pos_sampler = GlobalUniform(g, sample_size)
        
        self.neg_sampler = Time_NegativeSampler(k)

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, i):
        eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)
        
        src, dst = np.reshape(edges, (2, -1))
        relabeled_data = np.stack((src, rel, dst)).transpose()

        samples, labels = self.neg_sampler.sample(relabeled_data, num_nodes)

        import copy
        edge_labels = copy.deepcopy(labels)
        edge_labels[edge_labels != 0] = torch.tensor(self.g.edata['edge_labels'][eids].numpy()).to(labels.device)
        if 'diff_labels' in self.g.edata:
            diff_labels = copy.deepcopy(labels).long()
            diff_labels[diff_labels != 0] = torch.tensor(self.g.edata['diff_labels'][eids].numpy()).to(labels.device)

        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_rels))
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = torch.from_numpy(rel)
        sub_g.edata["norm"] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = torch.from_numpy(uniq_v).view(-1).long()

        if 'diff_labels' in self.g.edata:
            return sub_g, uniq_v, samples, labels, edge_labels, diff_labels
        return sub_g, uniq_v, samples, labels, edge_labels
