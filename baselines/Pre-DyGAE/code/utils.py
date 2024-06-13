import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, gmean
from torchmetrics.regression import TweedieDevianceScore
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, average_precision_score
import numpy as np

def get_subset_g(g, mask, num_rels, bidirected=False):
    src, dst = g.edges()
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata["etype"][mask]
    sub_edge_labels = g.edata['edge_labels'][mask]
    if 'diff_labels' in g.edata:
        sub_diff_labels = g.edata['diff_labels'][mask]

    if bidirected:
        sub_src, sub_dst = torch.cat([sub_src, sub_dst]), torch.cat(
            [sub_dst, sub_src]
        )
        sub_rel = torch.cat([sub_rel, sub_rel + num_rels])

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata[dgl.ETYPE] = sub_rel
    sub_g.edata['edge_labels'] = sub_edge_labels
    if 'diff_labels' in g.edata:
        sub_g.edata['diff_labels'] = sub_diff_labels
    return sub_g

def filter(
    triplets_to_filter, target_s, target_r, target_o, num_nodes,pos_num, skill_num, filter_o=True
):
    """Get candidate heads or tails to score"""
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]
    
    if filter_o:
        start_num = pos_num
        end_num = pos_num + skill_num
    else:
        start_num = 0
        end_num = pos_num
    for e in range(start_num, end_num):
        triplet = (
            (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        )
        
        if triplet not in triplets_to_filter:
            candidate_nodes = [e] + candidate_nodes
    return torch.LongTensor(candidate_nodes)
    

def perturb_and_get_filtered_rank(
    emb, w, s, r, o, test_size, triplets_to_filter, pos_num, skill_num, filter_o=True
):
    """Perturb subject or object in the triplets"""
    num_nodes = emb.shape[0]
    ranks = []
    for idx in tqdm(range(test_size)):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(
            triplets_to_filter,
            target_s,
            target_r,
            target_o,
            num_nodes,
            pos_num, skill_num,
            filter_o=filter_o,
        )
        if filter_o:
            emb_s = emb[target_s]
            emb_o = emb[candidate_nodes]
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = candidate_nodes.shape[0] - 1
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)

    return torch.LongTensor(ranks)



def calc_mrr(
    emb, w, test_mask, triplets_to_filter, batch_size=100, filter=True
):
    pos_num = triplets_to_filter[:,0].max() + 1
    skill_num = triplets_to_filter[:,2].max() - triplets_to_filter[:,0].max()
    with torch.no_grad():
        test_triplets = triplets_to_filter[test_mask]
        s, r, o = test_triplets[:, 0], test_triplets[:, 1], test_triplets[:, 2]
        test_size = len(s)
        triplets_to_filter = {
            tuple(triplet) for triplet in triplets_to_filter.tolist()
        }
        ranks_s = perturb_and_get_filtered_rank(
            emb, w, s, r, o, test_size, triplets_to_filter, pos_num, skill_num, filter_o=False
        )
        ranks_o = perturb_and_get_filtered_rank(
            emb, w, s, r, o, test_size, triplets_to_filter, pos_num, skill_num,
        )
        ranks_s += 1
        ranks_o += 1
        ranks = torch.cat([ranks_s, ranks_o])
        ranks = {"head": ranks_s, "tail": ranks_o, "all": ranks}
        results = {}
        for mode in ranks:
            mrr = torch.mean(1.0 / ranks[mode].float())
            hits1 = (ranks[mode] <= 1).sum() / ranks[mode].shape[0]
            hits3 = (ranks[mode] <= 3).sum() / ranks[mode].shape[0]
            hits10 = (ranks[mode] <= 10).sum() / ranks[mode].shape[0]
            results[mode] = {
                "mrr": mrr.item(),
                "hits1": hits1.item(),
                "hits3": hits3.item(),
                "hits10": hits10.item(),
            }

    return results

def pearson_correlation_coefficient(matrix1, matrix2):
    non_zero_idx_1 = matrix1.nonzero(as_tuple=False)
    non_zero_idx_2 = matrix2.nonzero(as_tuple=False)

    matrix1_non_zero_rows = matrix1[non_zero_idx_1[:, 0]]
    matrix2_non_zero_rows = matrix2[non_zero_idx_2[:, 0]]

    row_correlations = torch.stack([torch.nn.functional.cosine_similarity(matrix1_non_zero_rows[i], matrix2_non_zero_rows[i], dim=0) for i in range(matrix1_non_zero_rows.size(0))])

    mean_correlation = torch.mean(row_correlations)

    return mean_correlation.item()

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, y_true, y_pred):
        quantile = self.quantile
        prediction_underestimation = y_true - y_pred
        quantile_loss = torch.max(quantile * prediction_underestimation, (quantile - 1) * prediction_underestimation)
        return torch.mean(quantile_loss)

def error_geometric_mean(p, q):
    diff = torch.abs(p-q)
    geo_mean = gmean(diff.detach().cpu().numpy())
    return geo_mean

class PICP(nn.Module):
    def __init__(self, confidence_level=0.95):
        super(PICP, self).__init__()
        self.confidence_level = confidence_level

    def forward(self, predictions, true_targets):
        confidence_level = self.confidence_level
        prediction_mean = predictions.mean()
        prediction_variance = predictions.var()
        prediction_std = torch.sqrt(prediction_variance)

        
        half_width = torch.erfinv(torch.tensor(confidence_level)) * prediction_std * 2**0.5

        
        lower_bound = prediction_mean - half_width
        upper_bound = prediction_mean + half_width

        
        within_interval = torch.logical_and(true_targets >= lower_bound, true_targets <= upper_bound)
        
        coverage_probability = within_interval.float().mean().item()
        return coverage_probability

class MPIW(nn.Module):
    def __init__(self, confidence_level=0.95):
        super(MPIW, self).__init__()
        self.confidence_level = confidence_level

    def forward(self, predictions, true_targets):
        confidence_level = self.confidence_level
        prediction_mean = predictions.mean()
        prediction_variance = predictions.var()
        
        confidence_level = 0.95
        
        prediction_std = torch.sqrt(prediction_variance)
        
        half_width = torch.erfinv(torch.tensor(confidence_level)) * prediction_std * 2**0.5
        
        mean_width = torch.mean(half_width)
        return mean_width

class TweedieLoss(nn.Module):
    def __init__(self, power):
        super(TweedieLoss, self).__init__()
        self.power = power

    def forward(self, y_true, y_pred):
        
        y_pred = torch.clamp(y_pred, min=1e-6)
        
        loss = -(y_true * torch.log(y_pred) - y_pred ** self.power) / self.power
        return loss.mean()

class TweedieLossWithWeight(nn.Module):
    def __init__(self, power):
        super(TweedieLossWithWeight, self).__init__()
        self.power = power

    def forward(self, y_true, y_pred):
        
        y_pred = torch.clamp(y_pred, min=1e-6)
        
        
        loss = -(y_true * torch.log(y_pred) - y_pred ** self.power) / self.power
        
        
        weights = torch.where(y_true > 0.5, torch.tensor(2.0), torch.tensor(1.0))  
        weighted_loss = weights * loss

        return weighted_loss.mean()

def info_nce_loss(embeddings, labels, temperature=1.0):
    """
    Compute the InfoNCE loss for contrastive learning.

    Args:
        embeddings (torch.Tensor): NxD tensor containing the embeddings of N data points in D dimensions.
        labels (torch.Tensor): 1D tensor containing the labels of the data points.
        temperature (float, optional): Temperature parameter for the softmax. Default is 1.0.

    Returns:
        torch.Tensor: InfoNCE loss.
    """
    
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    
    positive_col_labels = labels.unsqueeze(1).expand(-1, embeddings.size(0))
    positive_row_labels = labels.T.unsqueeze(0).expand(embeddings.size(0), -1)
    positive_labels = positive_col_labels == positive_row_labels

    
    numerator = torch.exp(similarity_matrix / temperature)

    
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    
    loss = -torch.log(numerator / denominator.unsqueeze(-1))

    mask = torch.eye(embeddings.size(0), dtype=torch.bool)
    positive_labels[~mask] = False
    loss = loss[positive_labels].mean()

    return loss



def kl_div(log_std, mean, logits):
    kl_divergence = (0.5 / logits.size(0)* (1 + 2 * log_std - mean**2 - torch.exp(log_std) ** 2).sum(1).mean())
    return -kl_divergence



def rg_cal_mrr(pred, label):
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    M, N = pred.shape
    hits_at_n = np.zeros(N)
    mrr_sum = 0
    for user in range(M):
        for i in range(N):
            if label[user, i] == 1:
                import copy
                filter_score = copy.deepcopy(pred[user])
                filter_score[label[user].nonzero()[0]] = filter_score.min() - 1
                filter_score[i] = pred[user][i]
                ranked_indices = np.argsort(-filter_score)
                rank = np.where(ranked_indices == i)[0][0] + 1  
                mrr_sum += 1 / rank  
                for n in [1,3,5]:
                    if rank <= n:
                        hits_at_n[n] += 1
    mrr = mrr_sum / label.sum()
    metric = {
        "MRR": mrr,
        "Hits@1": hits_at_n[1] / label.sum(),
        "Hits@3": hits_at_n[3] / label.sum(),
    }
    return metric

def cal_auc(pred, label):
    pred, label = pred.view(-1), label.view(-1)
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    auc = roc_auc_score(label, pred)
    return auc

def cal_ap(pred, label):
    pred, label = pred.view(-1), label.view(-1)
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    ap = average_precision_score(label, pred)
    return ap

def cal_smape(pred, label):
    pred, label = pred.view(-1), label.view(-1)
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    denominator = (np.abs(label) + np.abs(pred)) / 2.0
    diff = np.abs(label - pred) / denominator
    diff[denominator == 0] = 0  
    smape = np.mean(diff)
    return smape

def time_metric(y, y_hat):
    label = torch.zeros(y.shape).to(y.device)
    label[y!=0] = 1
    label = label.long()
    
    metrics = rg_cal_mrr(y_hat, label)
    metrics['AUC'] = cal_auc(y_hat, label)
    metrics['AP'] = cal_ap(y_hat, label)
    metrics['SMAPE'] = cal_smape(y_hat, label)
    metrics['MAE'] = mean_absolute_error(y[y!=0].detach().cpu().numpy(), y_hat[y!=0].detach().cpu().numpy())
    metrics['RMSE'] = torch.sqrt(nn.MSELoss()(y[y!=0], y_hat[y!=0])).item()
    metrics['MAPE'] = mean_absolute_percentage_error(y[y!=0].detach().cpu().numpy(), y_hat[y!=0].detach().cpu().numpy())
    metrics['EGM'] = error_geometric_mean(y_hat[y!=0], y[y!=0]).item()
    return metrics


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_true, y_pred):
        
        weights = torch.where(y_true > 0.5, torch.tensor(2.0), torch.tensor(1.0))  
        loss = torch.mean(weights * (y_true - y_pred)**2)
        return loss

def compute_time_slice_similarity(embeddings):
    T, num, dim = embeddings.shape
    embeddings_flat = embeddings.reshape(T, -1)  
    sim = F.cosine_similarity(
        embeddings_flat[:, None, :], embeddings_flat[None, :, :], dim=2)
    return sim

def attraction_loss(embeddings, margin=2.0, temperature=2):
    sim = compute_time_slice_similarity(embeddings) / temperature
    mask = ~torch.eye(sim.size(0), dtype=torch.bool)  
    return (margin - sim[mask]).mean()

def repulsion_loss(embeddings, margin=2.0, temperature=2):
    sim = compute_time_slice_similarity(embeddings) / temperature
    mask = ~torch.eye(sim.size(0), dtype=torch.bool)  
    return (sim[mask] + margin).mean()