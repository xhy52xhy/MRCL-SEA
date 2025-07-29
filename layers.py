#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class MultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False, use="node"):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.use = use
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))

        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()

        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        if init is not None and diag:
            init(self.w)
            print(("w: ", self.w.shape))
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj, rel_co_weight=None):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            edge = adj._indices()
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))
            if self.use == "rel":
                edge_e = (edge_e + rel_co_weight)/2
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda() if next(self.parameters()).is_cuda else torch.ones(size=(N, 1))) # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)
            
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)
            
            output.append(h_prime.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


import torch
import torch.nn as nn
import math
from torch.nn import Parameter


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 主路径
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.l2(x)
        return x


class CNNAggregator(nn.Module):
    def __init__(self, word_embed_dim, num_filters, kernel_size, output_dim):
        super(CNNAggregator, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=word_embed_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.linear = nn.Linear(in_features=num_filters, out_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, seq_list):
        # padded_seq = pad_sequence(seq_list, batch_first=True)  # (ent_num, max_rel_num, rel_emb_dim)
        seq_list = seq_list.unsqueeze(0)    # (1, rel_num, rel_emb_dim)
        # print(seq_list.shape)
        # x: batch_size x max_sentence_len x word_embed_dim
        x = seq_list.permute(0, 2, 1)
        x = self.conv1d(x)  # batch_size x num_filters x max_sentence_len - kernel_size + 1
        # print("conv: ", x.shape)        x = self.relu(x)
        x = nn.functional.avg_pool1d(x, kernel_size=x.shape[-1])
        x = x.squeeze(-1)
        x = self.linear(x)

        return x


class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(0)  # inputs.shape = (batch_size, seq_len, input_dim)
        inputs = self.embedding(inputs)
        # inputs.shape = (batch_size, seq_len, hidden_dim)
        outputs, _ = self.gru(inputs)
        # outputs.shape = (batch_size, seq_len, hidden_dim * 2)
        outputs = self.linear(outputs)
        # outputs.shape = (batch_size, seq_len, hidden_dim)
        return outputs.mean(dim=1)
