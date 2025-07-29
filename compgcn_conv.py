import torch
import torch.nn as nn
# from helper import *
import numpy as np
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_geometric.nn.conv import MessagePassing
# from message_passing import MessagePassing
from torch_geometric.utils import softmax
# from torch_scatter import scatter
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, r_hidden, num_rels, num_ents, params=None, use_lg=False):
        super(self.__class__, self).__init__()
        self.is_bias = True
        self.p = params
        self.att_mode = 'learn'
        self.opn = 'cross'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r_hidden = r_hidden
        self.num_rels = num_rels
        self.num_ent = num_ents
        self.in_norm = None
        self.out_norm = None
        self.use_lg = use_lg
        self.is_bias = False
        self.act = nn.RReLU()
        self.device = None
        if self.is_bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.num_head = 1
        # if in_channels != r_hidden:
        self.trans_rel = nn.Linear(r_hidden, in_channels, bias=False)

        self.att_weight_1 = get_param((1, out_channels))
        self.w1_loop = get_param((in_channels, out_channels))
        self.w1_in = get_param((in_channels, out_channels))
        self.w1_out = get_param((in_channels, out_channels))
        # self.w_res = get_param((in_channels, out_channels))

        self.loop_rel = get_param((1, in_channels))
        self.w_rel = get_param((in_channels, r_hidden))
        self.drop = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.rel_weight1 = get_param((2 * self.num_rels + 1, in_channels))
        self.rel_weight2 = get_param((2 * self.num_rels + 1, in_channels))
        self.rel_weight3 = get_param((2 * self.num_rels + 1, in_channels))

        if self.num_head == 2:
            self.w2_loop = get_param((in_channels, out_channels))
            self.w2_in = get_param((in_channels, out_channels))
            self.w2_out = get_param((in_channels, out_channels))
            self.att_weight_2 = get_param((1, out_channels))
        if not self.use_lg:
            self.ra_layer = RALayer(e_hidden=in_channels, r_hidden=r_hidden)

    def forward(self, x, edge_index, edge_type, rel_embed):
        if self.device is None:
            self.device = edge_index.device
        # if self.in_channels != self.r_hidden:
    
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type, out_type = edge_type[:num_edges], edge_type[num_edges:]

        rel_embed = self.trans_rel(rel_embed)
        if not self.use_lg:
            rel_embed_in = self.ra_layer(x, in_index, in_type, rel_embed)
            # rel_embed_out = self.ra_layer(x, out_index, out_type, rel_embed)
            # rel_embed_out[:self.num_rels] = rel_embed_in
            # rel_embed_all = (rel_embed_in + rel_embed_out) / 2
            # rel_embed = torch.cat([rel_embed, rel_embed_out], dim=1)
        
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        in_res1 = self.propagate(in_index, x=x, edge_type=in_type, rel_embed=rel_embed, rel_weight=self.rel_weight1,
                                 edge_norm=self.in_norm, mode='in', w_str='1')
        loop_res1 = self.propagate(loop_index, x=x, edge_type=loop_type, rel_embed=rel_embed,
                                   rel_weight=self.rel_weight2, edge_norm=None, mode='loop', w_str='1')
        out_res1 = self.propagate(out_index, x=x, edge_type=out_type, rel_embed=rel_embed, rel_weight=self.rel_weight3,
                                  edge_norm=self.out_norm, mode='out', w_str='1')

        if self.num_head == 2:
            in_res2 = self.propagate(in_index, x=x, edge_type=in_type, rel_embed=rel_embed, rel_weight=self.rel_weight2,
                                     edge_norm=self.in_norm, mode='in', w_str='2')
            loop_res2 = self.propagate(loop_index, x=x, edge_type=loop_type, rel_embed=rel_embed,
                                       rel_weight=self.rel_weight2, edge_norm=None, mode='loop', w_str='2')
            out_res2 = self.propagate(out_index, x=x, edge_type=out_type, rel_embed=rel_embed,
                                      rel_weight=self.rel_weight2,
                                      edge_norm=self.out_norm, mode='out', w_str='2')
        if self.num_head == 2:
            out1 = in_res1 * (1 / 3) + out_res1 * (1 / 3) + loop_res1 * (1 / 3)
            out2 = in_res2 * (1 / 3) + out_res2 * (1 / 3) + loop_res2 * (1 / 3)
            out = 1 / 2 * (out1 + out2)
        else:
            # out = in_res1 * (1 / 3) + out_res1 * (1 / 3) + loop_res1 * (1 / 3)
            out = self.drop(in_res1) * (1 / 3) + self.drop(out_res1) * (1 / 3) + self.drop(loop_res1) * (1 / 3)

        if self.is_bias:
            out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed, rel_weight, edge_index_i, x_i):
        if self.att_mode == 'self_att':
            ent_embed = torch.mm(ent_embed, self.V)
        if self.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed) * rel_weight
        elif self.opn == 'no':
            trans_embed = ent_embed
        elif self.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.opn == 'mult':
            trans_embed = ent_embed * rel_embed * rel_weight
        elif self.opn == 'cross':
            trans_embed = ent_embed * rel_embed * rel_weight + ent_embed * rel_weight
        elif self.opn == 'concat':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1)
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, rel_embed, rel_weight, edge_norm, mode, w_str):
        # print(mode)
        weight = getattr(self, 'w{}_{}'.format(w_str, mode))
        att_w = getattr(self, 'att_weight_{}'.format(w_str))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        rel_weight_big = torch.index_select(rel_weight, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb, rel_weight_big, edge_index_i, x_i)

        if edge_norm is not None:
            edge_norm = torch.index_select(edge_norm, 0, edge_index_j)
        if self.opn == 'concat':
            xj_rel = xj_rel
        else:
            xj_rel = torch.mm(xj_rel, weight)

        x_i = torch.mm(x_i, weight)
        x_j = torch.mm(x_j, weight)
        rel_emb = torch.mm(rel_emb, weight)

        alpha = self._get_attention(edge_index_i, edge_index_j, edge_type, x_i, x_j, rel_emb, rel_weight_big, xj_rel,
                                    edge_norm, att_w)
        alpha = self.drop(alpha)

        xj_rel = xj_rel * alpha

        # xj_rel = torch.mm(xj_rel, weight)

        return xj_rel

    def _get_attention(self, edge_index_i, edge_index_j, edge_type, x_i, x_j, rel_embed, rel_weight, mes_xj, edge_norm,
                       att_multi):
        if self.att_mode == 'learn':
            alpha = self.leakyrelu(torch.einsum('ef, xf->e', [mes_xj, att_multi]))  # [E K]
            alpha = softmax(alpha, edge_index_i, num_nodes=self.num_ent)
        elif self.att_mode == 'sub_obj':
            sub_emb = x_i
            obj_emb = x_j
            alpha = self.leakyrelu(torch.einsum('ef,ef->e', [sub_emb, obj_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.num_ent)
        # alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        else:
            raise NotImplementedError

        return alpha.unsqueeze(1)

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_rels={}, num_head={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels, self.num_head)


class RALayer(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(RALayer, self).__init__()
        self.ww = nn.Linear(e_hidden+r_hidden, 1, bias=False)
        self.e2r = nn.Linear(e_hidden, r_hidden, bias=False)
        # self.e_in = nn.Linear(e_hidden+r_hidden, e_hidden, bias=False)
        # self.e_out = nn.Linear(e_hidden+r_hidden, e_hidden, bias=False)

    def forward(self, x, edge_index, edge_type, rel_emb):
        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]
        e_x = self.e2r(x)
        e_head = e_x[edge_index_i]
        e_rel = rel_emb[edge_type]
        # h_t = e_head + e_tail
        dp_att = torch.sum(e_head * e_rel, dim=-1)
        attention_weights = torch.softmax(dp_att, dim=-1)
        weighted_e_rel = e_rel * attention_weights.unsqueeze(dim=-1)  # 扩展 attention_weights 的维度
        x_r = global_add_pool(weighted_e_rel, edge_type, size=None)  # 使用 edge_type 进行聚合
        # x_r = scatter(e_rel * torch.unsqueeze(attention_weights, dim=-1), edge_type, dim=0, reduce='sum')

        return F.relu(x_r)