import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


from hypnettorch.mnets.mlp import MLP
from hypnettorch.hnets.mlp_hnet import HMLP
from typing import Optional, Tuple
from itertools import combinations

import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



class normal_mlp(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_mlp, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support


class PositionalEncoding(nn.Module):
    '''Positional Encoding for Transformer
    '''

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _get_masked_seq_last(seq: torch.Tensor, l: torch.Tensor, batch_first: bool = True):
    if batch_first:
        assert seq.size(0) == l.size(0)
    else:
        assert seq.size(1) == l.size(0)
    last = []
    for i, x in enumerate(l - 1):
        y = seq[i][x] if batch_first else seq[x][i]
        last.append(y)
    last_tensor = torch.stack(last)
    return last_tensor


def _repeat_seq_dim(seq: torch.Tensor, seq_len: int):
    return seq.unsqueeze(1).repeat(1, seq_len, 1)


def dense_adaptive_diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    '''Adaptive pooling with specified clustering number
    '''
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask
    # print(s.shape)
    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
    # clus_loss = 
    # temp = self.heterognn_2(temp,adj)
    # temp = self.heterognn_2(temp, adj)
    return out, out_adj, link_loss, ent_loss


class BaseSeqModel(nn.Module):

    '''Basic sequential model for prediction, providing default operation, modules and loss functions
    '''

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def default_amplifier(self, embed_dim):
        return nn.Linear(1, embed_dim)

    def default_decoder(self, embed_dim, hidden_dim, dropout, class_num):
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, class_num)
        )

    def forward(self):
        raise NotImplementedError

    def _amplify(self, x, amplifier):
        x = x.unsqueeze(-1)
        y = amplifier(x)
        return y

    def _encode(self):
        raise NotImplementedError

    def _decode(self, x, decoder):
        y = decoder(x)
        y = F.log_softmax(y, dim=-1)
        return y

class BaseRNNModel(BaseSeqModel):
    '''Override BaseSeqModel for recurrent neural networks variants, which can easily used based on this class
    '''
    # "type": "RNN",
    #         "args":{
    #             "embed_dim": 128,
    #             "class_num": 5,
    #             "hidden_dim": 32,
    #             "rnn_layer_num": 3,
    #             "dropout": 0.2
    #         }
    def __init__(
        self,
        ENCODER: nn.Module,
        embed_dim: int,
        rnn_layer_num: int,
        class_num: int,
        dropout: float,
        *args,
        **kwargs
    ):
        super().__init__()
        
        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.demand_encoder = ENCODER(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True
        )
        self.supply_encoder = ENCODER(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layer_num,
            batch_first=True
        )
        self.demand_decoder = self.default_decoder(embed_dim, embed_dim, dropout, class_num)
        self.supply_decoder = self.default_decoder(embed_dim, embed_dim, dropout, class_num)

    def forward(self, x, l, s, t_s, t_e, g, g_1,g_2, gap):
        d_x, s_x = x
        # print(s_x.shape)
        l  = l.expand(d_x.shape[0])
        # print(d_x.shape)
        d_x = self._amplify(d_x, self.demand_amplifier)
        s_x = self._amplify(s_x, self.supply_amplifier)
        # print(d_x.shape)
        d_x = self._encode(d_x, l, self.demand_encoder)
        s_x = self._encode(s_x, l, self.supply_encoder)
        d_y = self._decode(d_x, self.demand_decoder)
        s_y = self._decode(s_x, self.supply_decoder)
        return (d_y, s_y) , 0, 0, 0

    def _encode(self, x, l, encoder):
        
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = encoder(x)
        y, _ = pad_packed_sequence(y, batch_first=True)
        # y = _get_masked_seq_last(y, l)
        y = y[:, -1,:]
        return y
    

class RNN(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.RNN, *args, **kwargs)


class GRU(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.GRU, *args, **kwargs)


class LSTM(BaseRNNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(ENCODER=nn.LSTM, *args, **kwargs)

class Transformer(BaseSeqModel):
    def __init__(self, skill_num, embed_dim, skill_embed_dim, class_num, nhead, rnn_layer_num, dropout, model, *args, **kwargs):
        super().__init__()
        self.src_mask = None
        self.embed_dim = embed_dim
        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.position_encoder = PositionalEncoding(embed_dim, dropout)
        self.demand_transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, embed_dim, dropout), rnn_layer_num)
        self.supply_transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead, embed_dim, dropout), rnn_layer_num)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.end_token = nn.Embedding(1, skill_embed_dim)
        self.demand_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, class_num),
            )
        self.supply_MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, class_num),
            ) 
    def forward(self, x, l, s, t_s, t_e, g, comm, skill_semantic_embed, gap):
        d_x, s_x = x
        d_x = d_x[:,:l]
        s_x = s_x[:,:l]
        amp_d_x = self._amplify(d_x, self.demand_amplifier)
        amp_s_x = self._amplify(s_x, self.supply_amplifier)
        amp_d_x = torch.cat([amp_d_x,self.end_token.weight.expand(amp_d_x.size(0),1, -1)],dim=1)
        amp_s_x = torch.cat([amp_s_x,self.end_token.weight.expand(amp_s_x.size(0),1, -1)],dim=1)
        enc_d_x = self._encode_Trans(amp_d_x, l, self.position_encoder, self.demand_transformer_encoder)
        enc_s_x = self._encode_Trans(amp_s_x, l, self.position_encoder, self.supply_transformer_encoder)
        d_y, s_y  = self._decode((enc_d_x[:,-1,:],enc_s_x[:,-1,:]))
        
        return (d_y, s_y), 0, 0, 0
    def _encode_Trans(self, x, l, position_encoder, transformer_encoder, has_mask=True):
        x = x.permute(1, 0, 2)
        # x = torch.cat([init_emb.unsqueeze(0),x],dim=0)
        # mask
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = _generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        # encode

        x = x * math.sqrt(self.embed_dim)
        x = position_encoder(x)
        y = transformer_encoder(x, self.src_mask)
        # y = _get_masked_seq_last(y, l, batch_first=False)
        y = self.linear(y)
        y = y.permute(1, 0, 2)
        return y
    def _decode(self, x):
        d_x, s_x = x
        d_y = self.demand_MLP(d_x)
        s_y = self.supply_MLP(s_x)
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)
# ----------------------Evolve Graph Models----------------------


class Model(BaseSeqModel):
    '''Dynamic Heterogeneous Graph Enhanced Meta-learner
    '''
    # def __init__(self, configs, individual=False):
    def __init__(self, configs=None, skill_num=16, embed_dim=768, skill_embed_dim=768, class_num=1, nhead=2, nlayers=1, dropout=0.1, model='normal', hyperdecode=False, hypcomb="none",gcn_layers=2,delta=0.1):
        super().__init__()
        if configs is not None:
            skill_num = configs.enc_in
        self.pred_len = configs.pred_len

        self.src_mask = None
        self.embed_dim = embed_dim
        self.nlayer = nlayers
        self.gcn_layers = gcn_layers
        self.model = model
        self.hyper_size = int(embed_dim/4)
        self.hyperdecode = hyperdecode
        self.hyper_n_layers = 1

        # amplifier
        self.demand_amplifier = self.default_amplifier(embed_dim)
        self.supply_amplifier = self.default_amplifier(embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)
        # encoder
        self.position_encoder = PositionalEncoding(embed_dim, dropout)
        self.demand_LSTM_encoder = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,
                                    num_layers=nlayers, batch_first=True, dropout=dropout)
        self.supply_LSTM_encoder = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,
                                    num_layers=nlayers, batch_first=True, dropout=dropout)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.init_skill_emb = nn.Embedding(skill_num, skill_embed_dim)
        self.init_start_token = nn.Embedding(1, skill_embed_dim)
        
        lin_setting = (2,3)
        if model in ["static","adaptive"]:
            lin_setting = (3,2)
        # decoder
        self.desupply_merge = nn.Linear(embed_dim * lin_setting[0], embed_dim)
        self.demand_MLP = nn.Sequential(
            nn.Linear(embed_dim* lin_setting[1], embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, self.pred_len)
            )
        self.supply_MLP = nn.Sequential(
            nn.Linear(embed_dim* lin_setting[1], embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, self.pred_len)
            ) 
        if hyperdecode == True:
            self.before_hyp_MLP = nn.Sequential(
                nn.Linear(embed_dim*3, embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                )
            self.demand_MLP = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, self.pred_len)
                )
            self.supply_MLP = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, self.pred_len)
                ) 
        self.hypcomb = hypcomb
        
        if hyperdecode == True:
            self.condition = torch.rand(embed_dim*2)
            self.gap_encoder = nn.LSTM(input_size=1,hidden_size=embed_dim,
                                        num_layers=nlayers, batch_first=True)
            self.main_MLP = MLP(n_in=embed_dim*3, n_out=embed_dim, hidden_layers=[embed_dim for _ in range(2)], dropout_rate=0.1, no_weights=False,use_batch_norm=True)
            print("parameter :", self.main_MLP.param_shapes)
            self.hyper_MLP = HMLP(self.main_MLP.param_shapes, uncond_in_size=0, cond_in_size=skill_embed_dim*2,
                layers=(embed_dim,embed_dim), num_cond_embs=1, use_batch_norm=True)
            self.norm_cond = nn.LayerNorm(embed_dim*2)
            self.norm_input = nn.LayerNorm(embed_dim)
            self.hyper_attention = torch.nn.MultiheadAttention(skill_embed_dim,num_heads=1,add_zero_attn=True,batch_first=True)
            self.skill_type_emb = nn.Embedding(4, skill_embed_dim)
            self.supply_demand_task_emb = nn.Embedding(2, skill_embed_dim)

        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # print(gap.shape)
        '''
        Graph Evolve
        1. input
        '''
        x_enc = x_enc.permute(0,2,1)
        bs, skill_num, dim = x_enc.shape
        d_x, s_x = x_enc, x_enc  # [batch_size, 18]
        d_x = d_x.reshape(-1, d_x.shape[-1])
        s_x = s_x.reshape(-1, d_x.shape[-1])

        l  = d_x.shape[-1] * torch.ones([d_x.shape[0]]).long()

        embedding = self.init_skill_emb.weight
        embedding = embedding.repeat(bs*2,1)
        amp_d_x = self._amplify(d_x, self.demand_amplifier) # [batch_size, 18, 128]
        amp_s_x = self._amplify(s_x, self.supply_amplifier)

        start_token = self.init_start_token.weight.expand(d_x.shape[0],-1).unsqueeze(1)
        enc_d_x_seq = self._encode(amp_d_x, l, embedding, self.position_encoder, self.demand_LSTM_encoder) # [batch_size, l, 128]
        enc_s_x_seq = self._encode(amp_s_x, l, embedding, self.position_encoder, self.supply_LSTM_encoder)

        loss = 0


        enc_d_x = enc_d_x_seq[:,-1,:]
        enc_s_x = enc_s_x_seq[:,-1,:]

        if self.hyperdecode:
            enc_gap = gap.squeeze().unsqueeze(-1)
            enc_gap = self._encode(enc_gap, l, self.init_skill_emb.weight, self.position_encoder, self.gap_encoder)
            d_y, s_y = self._hyperdecode((enc_d_x, enc_s_x), embedding, enc_gap[:,-1,:]) #[batch_size, 10]
        elif self.model in ["static", "adaptive"]:
            d_y, s_y = self._combined_decode((enc_d_x, enc_s_x), embedding) #[batch_size, 10]
        else:
            d_y, s_y = self._decode((enc_d_x, enc_s_x), embedding) #[batch_size, 10]
        
        learned_graph = 0
        pred = 0

        d_y = d_y.reshape(bs,skill_num,self.pred_len).permute(0,2,1)
        d_y = nn.LeakyReLU()(d_y)
        return d_y

    def _combined_amplify(self, x, s, amplifier_1, amplifier_2):
        x = x.unsqueeze(-1)
        y = F.leaky_relu(amplifier_1(x))
        y = torch.concat([y, s], dim=-1)
        y = F.leaky_relu(amplifier_2(y))
        return y
    def _amplify(self, x, amplifier_1):
        x = x.unsqueeze(-1)
        y = F.leaky_relu(amplifier_1(x))
        return y

    def _encode_Trans(self, x, l, init_emb, position_encoder, transformer_encoder, has_mask=True):
        x = x.permute(1, 0, 2)

        # mask
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = _generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        # encode

        x = x * math.sqrt(self.embed_dim)
        x = position_encoder(x)
        y = transformer_encoder(x, self.src_mask)
        y = y.permute(1, 0, 2)
        return y
    def _encode(self, x, l, init_emb, position_encoder, encoder, has_mask=True):
        x = pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = encoder(x) 
        y, _ = pad_packed_sequence(y, batch_first=True)

        return y
    def _combined_decode(self, x, s):
        d_x, s_x = x
        d_s = self.drop(self.desupply_merge(torch.concat([d_x, s_x, s], dim=-1)))
        d_y = self.demand_MLP(torch.concat([d_x, d_s[:,:]], dim=-1))
        s_y = self.supply_MLP(torch.concat([s_x, d_s[:,:]], dim=-1))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)
    def _decode(self, x, s):
        d_x, s_x = x
        d_s = self.drop(self.desupply_merge(torch.concat([d_x, s_x], dim=-1)))
        d_y = self.demand_MLP(torch.concat([d_x, s[:d_x.shape[0],:], d_s], dim=-1))
        s_y = self.supply_MLP(torch.concat([s_x, s[d_x.shape[0]:, :], d_s], dim=-1))
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)
    def _hyperdecode(self, x, s, cond=None):
        d_x, s_x = x
        x_in = torch.cat([d_x,s_x],dim=0)
        d_s = self.desupply_merge(torch.concat([d_x, s_x], dim=-1))
        sd_emb = self.supply_demand_task_emb(torch.cat([d_x.new_zeros(d_x.shape[0]), d_x.new_ones(d_x.shape[0])],dim=0).to(d_x.device).type(torch.int32))
        if self.hypcomb == "none":
            # print("here")
            d_y = self.demand_MLP(self.before_hyp_MLP(torch.concat([d_x, s[:d_x.shape[0],:]], dim=-1)))
            s_y = self.supply_MLP(self.before_hyp_MLP(torch.concat([s_x, s[d_x.shape[0]:, :]], dim=-1)))
            return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)
        hyper_feat_d = torch.cat([s[:d_x.shape[0],:], cond],dim=-1)
        hyper_feat_s = torch.cat([s[d_x.shape[0]:,:], cond],dim=-1)
        
        # self.condition = self.condition.to(hyper_feat_d.device)
        # hyper_feat_d = self.condition.expand(d_x.shape[0],-1)
        # hyper_feat_s = self.condition.expand(d_x.shape[0],-1)
        hyper_feat_d = self.norm_cond(hyper_feat_d)
        hyper_feat_s = self.norm_cond(hyper_feat_s)
        weight_d = self.hyper_MLP(cond_input=hyper_feat_d)
        weight_s = self.hyper_MLP(cond_input=hyper_feat_s)
        d_y_h = self.main_MLP(torch.concat([d_x, s[:d_x.shape[0],:], d_s], dim=-1), weight_d[0])
        s_y_h = self.main_MLP(torch.concat([s_x, s[d_x.shape[0]:,:], d_s], dim=-1), weight_s[0])
        
        if self.hypcomb == "res":
            d_y = d_x + d_y_h
            s_y = d_x + s_y_h
            d_y = self.demand_MLP(d_y)
            s_y = self.supply_MLP(s_y)
        elif self.hypcomb == 'add':
            d_y = self.before_hyp_MLP(torch.concat([d_x, s[:d_x.shape[0],:], d_s], dim=-1))
            s_y = self.before_hyp_MLP(torch.concat([s_x, s[d_x.shape[0]:,:], d_s], dim=-1))
            d_y = d_y + d_y_h
            s_y = s_y + s_y_h
            d_y = self.demand_MLP(d_y)
            s_y = self.supply_MLP(s_y)
        elif self.hypcomb == 'pool':
            d_y = self.before_hyp_MLP(torch.concat([d_x, s[:d_x.shape[0],:]], dim=-1))
            s_y = self.before_hyp_MLP(torch.concat([s_x, s[d_x.shape[0]:,:]], dim=-1))
            d_y = self.demand_MLP(d_y) + self.demand_MLP(d_y_h)
            s_y = self.supply_MLP(s_y) + self.demand_MLP(s_y_h)
        elif self.hypcomb == 'lambda':
            lamb=0.2
            d_y = self.before_hyp_MLP(torch.concat([d_x, s[:d_x.shape[0],:]], dim=-1))
            s_y = self.before_hyp_MLP(torch.concat([s_x, s[d_x.shape[0]:,:]], dim=-1))
            d_y = (1-lamb)*d_y + lamb*d_y_h
            s_y = (1-lamb)*s_y + lamb*s_y_h
            d_y = self.demand_MLP(d_y)
            s_y = self.supply_MLP(s_y)
        elif self.hypcomb == "pure":
            d_y = self.demand_MLP(d_y_h)
            s_y = self.supply_MLP(s_y_h)
        return F.log_softmax(d_y, dim=-1), F.log_softmax(s_y, dim=-1)

if __name__ == '__main__':
    skill_num = 35
    model = Skill_Evolve_Hetero(skill_num=skill_num)
    x = torch.randn([4, 6, skill_num])
    y = model(x)
    print(y.shape)